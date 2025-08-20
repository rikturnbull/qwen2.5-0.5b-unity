using UnityEngine;
using Unity.InferenceEngine;
using FF = Unity.InferenceEngine.Functional;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using Debug = UnityEngine.Debug;
using System.Threading.Tasks;
using System;
using UnityEngine.Networking;

public class ModelQwen
{
    private Qwen2Tokenizer _tokenizer;
    private Worker _engine;
    private Worker _decodeEngine;

    private const BackendType BACKEND = BackendType.CPU;
    private const int MAX_LAYERS = 24;
    private const int NUM_KEY_VALUE_HEADS = 2;
    private const int HEAD_DIM = 64;
    private const int TOKEN_IM_END = 151645;

    private List<int> _tokens = new();
    private List<int> _outputTokens = new();

    private int _maxNewTokens = 100;
    private List<int> _eosTokens;

    string systemMessage = "You are a helpful assistant. Answer concisely and clearly in up to three sentences.";

    private async Task<T> LoadStreamingAssetAsync<T>(string fileName)
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, fileName);

        using UnityWebRequest request = UnityWebRequest.Get(filePath);

        await request.SendWebRequest();

        if (request.result != UnityWebRequest.Result.Success)
        {
            Debug.LogError($"Failed to load {fileName}: {request.error}");
            throw new System.Exception($"Failed to load {fileName}: {request.error}");
        }

        if (typeof(T) == typeof(byte[]))
        {
            Debug.Log($"Successfully loaded {fileName} ({request.downloadedBytes} bytes)");
            return (T)(object)request.downloadHandler.data;
        }
        else if (typeof(T) == typeof(string))
        {
            Debug.Log($"Successfully loaded {fileName} ({request.downloadHandler.text.Length} characters)");
            return (T)(object)request.downloadHandler.text;
        }
        else
        {
            throw new ArgumentException($"Unsupported type {typeof(T)}. Only byte[] and string are supported.");
        }
    }

    public async Task Start()
    {
        var modelDataTask = LoadStreamingAssetAsync<byte[]>("qwen2.5-0.5b_uint8.sentis");
        var vocabContentTask = LoadStreamingAssetAsync<string>("vocab.json");
        var mergesContentTask = LoadStreamingAssetAsync<string>("merges.txt");
        var configContentTask = LoadStreamingAssetAsync<string>("tokenizer_config.json");

        byte[] modelData = await modelDataTask;
        string vocabContent = await vocabContentTask;
        string mergesContent = await mergesContentTask;
        string configContent = await configContentTask;

        _tokenizer = new Qwen2Tokenizer(vocabContent, mergesContent, configContent);
        _eosTokens = new List<int> { 151645, 151643 };

        using var modelStream = new MemoryStream(modelData);
        Model baseModel = ModelLoader.Load(modelStream);

        var vocab_size = 151936;
        FunctionalGraph graph = new FunctionalGraph();
        FunctionalTensor logitsInput = graph.AddInput<float>(new DynamicTensorShape(1, -1, vocab_size));
        FunctionalTensor argMax = FF.ArgMax(logitsInput, 2, false);
        Model greedyModel = graph.Compile(argMax);

        _engine = new Worker(baseModel, BACKEND);
        _decodeEngine = new Worker(greedyModel, BACKEND);

        await Warmup();
    }

    private async Task Warmup()
    {
        Debug.Log("Warming up the model...");
        var stopwatch = new Stopwatch();
        stopwatch.Start();

        using (var dummyInput = new Tensor<int>(new TensorShape(1, 1), new[] { 1 }))
        using (var dummyAttentionMask = new Tensor<int>(new TensorShape(1, 1), new[] { 1 }))
        using (var dummyPositionIds = new Tensor<int>(new TensorShape(1, 1), new[] { 0 }))
        {
            _engine.SetInput("input_ids", dummyInput);
            _engine.SetInput("attention_mask", dummyAttentionMask);
            _engine.SetInput("position_ids", dummyPositionIds);

            var emptyPastShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
            using (var emptyPastTensor = new Tensor<float>(emptyPastShape))
            {
                for (int i = 0; i < MAX_LAYERS; i++)
                {
                    _engine.SetInput($"past_key_values.{i}.key", emptyPastTensor);
                    _engine.SetInput($"past_key_values.{i}.value", emptyPastTensor);
                }
            }

            await ScheduleAsync(_engine);
            using var dummyLogits = _engine.PeekOutput("logits").ReadbackAndClone() as Tensor<float>;

            _decodeEngine.SetInput(0, dummyLogits);
            _decodeEngine.Schedule();
        }

        stopwatch.Stop();
        Debug.Log($"Warmup complete in {stopwatch.ElapsedMilliseconds} ms.");
    }

    public async Task Generate(string inputText, Action<string> callback)
    {
        string finalPrompt = $"<|im_start|>system\n{systemMessage}<|im_end|>\n<|im_start|>user\n{inputText}<|im_end|>\n<|im_start|>assistant\n";
        Debug.Log("Prompt: " + finalPrompt);

        _tokens.Clear();
        _tokens.AddRange(_tokenizer.Encode(finalPrompt));
        _outputTokens.Clear();

        var stopwatch = new Stopwatch();
        stopwatch.Start();

        int step = 0;
        int initialTokenCount = _tokens.Count;
        int[] initialTokens = _tokens.ToArray();
        int prefillSequenceLength = initialTokens.Length;

        using var inputTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), initialTokens);
        using var attentionMaskTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), Enumerable.Repeat(1, prefillSequenceLength).ToArray());
        using var positionIdsTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), Enumerable.Range(0, prefillSequenceLength).ToArray());

        _engine.SetInput("input_ids", inputTensor);
        _engine.SetInput("attention_mask", attentionMaskTensor);
        _engine.SetInput("position_ids", positionIdsTensor);

        var emptyPastShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
        Tensor<float>[] pastKeys = new Tensor<float>[MAX_LAYERS];
        Tensor<float>[] pastValues = new Tensor<float>[MAX_LAYERS];

        for (int i = 0; i < MAX_LAYERS; i++)
        {
            pastKeys[i] = new Tensor<float>(emptyPastShape);
            pastValues[i] = new Tensor<float>(emptyPastShape);

            _engine.SetInput($"past_key_values.{i}.key", pastKeys[i]);
            _engine.SetInput($"past_key_values.{i}.value", pastValues[i]);
        }

        await ScheduleAsync(_engine);

        using var outputLogits = _engine.PeekOutput("logits") as Tensor<float>;
        int nextToken = await ProcessLogits(outputLogits, prefillSequenceLength - 1);

        if (nextToken != TOKEN_IM_END)
        {
            _tokens.Add(nextToken);
            _outputTokens.Add(nextToken);
        }

        callback?.Invoke(_tokenizer.Decode(_outputTokens));

        step = 1;
        while (step < _maxNewTokens && !_eosTokens.Contains(nextToken))
        {
            for (int i = 0; i < MAX_LAYERS; i++)
            {
                pastKeys[i] = await _engine.PeekOutput($"present.{i}.key").ReadbackAndCloneAsync() as Tensor<float>;
                pastValues[i] = await _engine.PeekOutput($"present.{i}.value").ReadbackAndCloneAsync() as Tensor<float>;

                _engine.SetInput($"past_key_values.{i}.key", pastKeys[i]);
                _engine.SetInput($"past_key_values.{i}.value", pastValues[i]);
            }

            int currentSequenceLength = initialTokenCount + step;

            using var newInputTensor = new Tensor<int>(new TensorShape(1, 1), new[] { nextToken });
            using var newPositionIdsTensor = new Tensor<int>(new TensorShape(1, 1), new[] { currentSequenceLength - 1 });
            using var newAttentionMaskTensor = new Tensor<int>(new TensorShape(1, currentSequenceLength), Enumerable.Repeat(1, currentSequenceLength).ToArray());

            _engine.SetInput("input_ids", newInputTensor);
            _engine.SetInput("attention_mask", newAttentionMaskTensor);
            _engine.SetInput("position_ids", newPositionIdsTensor);

            await ScheduleAsync(_engine);

            using var newOutputLogits = _engine.PeekOutput("logits") as Tensor<float>;
            nextToken = await ProcessLogits(newOutputLogits, 0);

            if (nextToken != TOKEN_IM_END)
            {
                _tokens.Add(nextToken);
                _outputTokens.Add(nextToken);
            }

            callback?.Invoke(_tokenizer.Decode(_outputTokens));

            step++;
        }

        for (int i = 0; i < MAX_LAYERS; i++)
        {
            pastKeys[i]?.Dispose();
            pastValues[i]?.Dispose();
        }

        string generatedText = _tokenizer.Decode(_outputTokens);
        Debug.Log($"Final sequence: {generatedText}");

        stopwatch.Stop();
        Debug.Log($"<color=cyan><b>Total Generation Time: {stopwatch.ElapsedMilliseconds} ms</b></color>");

        callback?.Invoke(generatedText);
    }

    private async Task ScheduleAsync(Worker worker, int layersPerFrame = 75)
    {
        var schedule = worker.ScheduleIterable();
        int layersProcessed = 0;

        while (schedule.MoveNext())
        {
            layersProcessed++;
            if (layersProcessed % layersPerFrame == 0)
            {
                await Task.Yield();
            }
        }
    }
    private async Task<int> ProcessLogits(Tensor<float> logits, int sequenceIndex)
    {
        _decodeEngine.SetInput(0, logits);
        _decodeEngine.Schedule();
        using var argMaxTensor = await _decodeEngine.PeekOutput().ReadbackAndCloneAsync() as Tensor<int>;
        int nextToken = argMaxTensor.DownloadToArray()[sequenceIndex];
        return nextToken;
    }

    private void OnDestroy()
    {
        _engine?.Dispose();
        _decodeEngine?.Dispose();
    }
}
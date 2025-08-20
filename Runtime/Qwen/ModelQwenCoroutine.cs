// using UnityEngine;
// using Unity.InferenceEngine;
// using FF = Unity.InferenceEngine.Functional;
// using System.IO;
// using System.Linq;
// using System.Collections;
// using System.Collections.Generic;
// using UnityEngine.UI;
// using System.Diagnostics;
// using Debug = UnityEngine.Debug;
// using System.Threading.Tasks;
// using System;

// public class ModelQwenCoroutine
// {
//     private Qwen2Tokenizer _tokenizer;
//     private Worker _engine;
//     private Worker _decodeEngine;

//     private const BackendType BACKEND = BackendType.CPU;
//     private const int MAX_LAYERS = 24;
//     private const int NUM_KEY_VALUE_HEADS = 2;
//     private const int HEAD_DIM = 64;
//     private const int TOKEN_IM_END = 151645;
//     private const int ENGINE_ITERATIONS = 50;

//     private List<int> _tokens = new();
//     private List<int> _outputTokens = new();

//     private int _maxNewTokens = 100;
//     private List<int> _eosTokens;

//     // [SerializeField]
//     // Text textLabel;
//     string systemMessage = "You are a helpful assistant. Answer concisely and clearly in up to three sentences.";

//     public void Start()
//     {
//         string modelFilePath = Path.Combine(Application.streamingAssetsPath, "qwen2.5-0.5b_uint8.sentis");
//         string vocabFilePath = Path.Combine(Application.streamingAssetsPath, "vocab.json");
//         string mergeFilePath = Path.Combine(Application.streamingAssetsPath, "merges.txt");
//         string configFilePath = Path.Combine(Application.streamingAssetsPath, "tokenizer_config.json");

//         string vocabContent = File.ReadAllText(vocabFilePath);
//         string mergesContent = File.ReadAllText(mergeFilePath);
//         string configContent = File.ReadAllText(configFilePath);

//         _tokenizer = new Qwen2Tokenizer(vocabContent, mergesContent, configContent);
//         _eosTokens = new List<int> { 151645, 151643 };

//         Model baseModel = ModelLoader.Load(modelFilePath);

//         var vocab_size = 151936;
//         FunctionalGraph graph = new FunctionalGraph();
//         FunctionalTensor logitsInput = graph.AddInput<float>(new DynamicTensorShape(1, -1, vocab_size));
//         FunctionalTensor argMax = FF.ArgMax(logitsInput, 2, false);
//         Model greedyModel = graph.Compile(argMax);
//         Debug.Log($"Qwen:Greedy model output name: {greedyModel.outputs[0].name}");
//         _engine = new Worker(baseModel, BACKEND);
//         _decodeEngine = new Worker(greedyModel, BACKEND);

//         Warmup();
//     }

//     private void Warmup()
//     {
//         Debug.Log("Warming up the model...");
//         var stopwatch = new Stopwatch();
//         stopwatch.Start();

//         using (var dummyInput = new Tensor<int>(new TensorShape(1, 1), new[] { 1 }))
//         using (var dummyAttentionMask = new Tensor<int>(new TensorShape(1, 1), new[] { 1 }))
//         using (var dummyPositionIds = new Tensor<int>(new TensorShape(1, 1), new[] { 0 }))
//         {
//             _engine.SetInput("input_ids", dummyInput);
//             _engine.SetInput("attention_mask", dummyAttentionMask);
//             _engine.SetInput("position_ids", dummyPositionIds);

//             var emptyPastShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
//             using (var emptyPastTensor = new Tensor<float>(emptyPastShape))
//             {
//                 for (int i = 0; i < MAX_LAYERS; i++)
//                 {
//                     _engine.SetInput($"past_key_values.{i}.key", emptyPastTensor);
//                     _engine.SetInput($"past_key_values.{i}.value", emptyPastTensor);
//                 }
//             }

//             _engine.Schedule();
//             using var dummyLogits = _engine.PeekOutput("logits").ReadbackAndClone() as Tensor<float>;

//             _decodeEngine.SetInput(0, dummyLogits);
//             _decodeEngine.Schedule();
//         }

//         stopwatch.Stop();
//         Debug.Log($"Warmup complete in {stopwatch.ElapsedMilliseconds} ms.");
//     }

//     public IEnumerator Generate(String inputText, Dictionary<string, string> options, Action<string> callback)
//     {
//         string finalPrompt = $"<|im_start|>system\n{systemMessage}<|im_end|>\n<|im_start|>user\n{inputText}<|im_end|>\n<|im_start|>assistant\n";
//         Debug.Log("Prompt: " + finalPrompt);

//         // textLabel.text = "";

//         _tokens.Clear();
//         _tokens.AddRange(_tokenizer.Encode(finalPrompt));
//         _outputTokens.Clear();

//         var stopwatch = new Stopwatch();
//         stopwatch.Start();

//         int step = 0;
//         int initialTokenCount = _tokens.Count;
//         int[] initialTokens = _tokens.ToArray();
//         int prefillSequenceLength = initialTokens.Length;

//         Debug.Log($"Qwen:1:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//         using var inputTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), initialTokens);
//         Debug.Log($"Qwen:2:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//         using var attentionMaskTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), Enumerable.Repeat(1, prefillSequenceLength).ToArray());
//         Debug.Log($"Qwen:3:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//         using var positionIdsTensor = new Tensor<int>(new TensorShape(1, prefillSequenceLength), Enumerable.Range(0, prefillSequenceLength).ToArray());

//         _engine.SetInput("input_ids", inputTensor);
//         _engine.SetInput("attention_mask", attentionMaskTensor);
//         _engine.SetInput("position_ids", positionIdsTensor);

//         Debug.Log($"Qwen:4:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//         var emptyPastShape = new TensorShape(1, NUM_KEY_VALUE_HEADS, 0, HEAD_DIM);
//         Tensor<float>[] pastKeys = new Tensor<float>[MAX_LAYERS];
//         Tensor<float>[] pastValues = new Tensor<float>[MAX_LAYERS];
//         Debug.Log($"Qwen:5:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         for (int i = 0; i < MAX_LAYERS; i++)
//         {
//             pastKeys[i] = new Tensor<float>(emptyPastShape);
//             pastValues[i] = new Tensor<float>(emptyPastShape);

//             _engine.SetInput($"past_key_values.{i}.key", pastKeys[i]);
//             _engine.SetInput($"past_key_values.{i}.value", pastValues[i]);
//         }

//         Debug.Log($"Qwen:6:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         var iterator = _engine.ScheduleIterable();

//         int iterationCount = 0;
//         while (iterator.MoveNext())
//         {
//             iterationCount++;

//             // Yield every ENGINE_ITERATIONS iterations to keep Unity responsive
//             if (iterationCount % ENGINE_ITERATIONS == 0)
//             {
//                 yield return null;
//             }
//         }

//         Debug.Log($"Qwen:7:{iterationCount}Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         // using var outputLogits = _engine.PeekOutput("logits") as Tensor<float>;
//         Tensor<float> outputLogits = null;
//         yield return ReadOutputCoroutine(_engine, $"logits", (tensor) => outputLogits = tensor as Tensor<float>);

//         Debug.Log($"Qwen:8:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         int nextToken = 0;
//         yield return ProcessLogits(outputLogits, prefillSequenceLength - 1, (token) => nextToken = token);
//         // yield return new WaitUntil(() => task.IsCompleted);

//         // int nextToken = task.Result;

//         Debug.Log($"Qwen:9:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         if (nextToken != TOKEN_IM_END)
//         {
//             _tokens.Add(nextToken);
//             _outputTokens.Add(nextToken);
//         }

//         Debug.Log($"Qwen:10:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         // textLabel.text = _tokenizer.Decode(_outputTokens);
//         callback(_tokenizer.Decode(_outputTokens));

//         step = 1;
//         while (step < _maxNewTokens && !_eosTokens.Contains(nextToken))
//         {
//             Debug.Log($"Qwen:101:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//                 // // Read all outputs first
//                 // var keyTasks = new List<(int layer, Tensor tensor)>();
//                 // var valueTasks = new List<(int layer, Tensor tensor)>();
//                 // Debug.Log($"Qwen:102:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//                 // for (int i = 0; i < MAX_LAYERS; i++)
//                 // {
//                 //     Debug.Log($"Qwen:103:{step}:{i}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//                 //     var keyBuffer = _engine.PeekOutput($"present.{i}.key");
//                 //     var valueBuffer = _engine.PeekOutput($"present.{i}.value");
                    
//                 //     keyBuffer.ReadbackRequest();
//                 //     valueBuffer.ReadbackRequest();
                    
//                 //     keyTasks.Add((i, keyBuffer));
//                 //     valueTasks.Add((i, valueBuffer));
//                 // }
//                 // Debug.Log($"Qwen:104:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//                 // // Wait for all readbacks to complete
//                 // bool allComplete = false;
//                 // while (!allComplete)
//                 // {
//                 //     Debug.Log($"Qwen:105:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//                 //     allComplete = true;
//                 //     foreach (var (layer, tensor) in keyTasks.Concat(valueTasks))
//                 //     {
//                 //         if (!tensor.IsReadbackRequestDone())
//                 //         {
//                 //             allComplete = false;
//                 //             break;
//                 //         }
//                 //     }
//                 //     if (!allComplete) yield return null;
//                 // }

//                 // // Process results
//                 // for (int i = 0; i < MAX_LAYERS; i++)
//                 // {
//                 //     Debug.Log($"Qwen:106:{step}:{i}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//                 //     pastKeys[i] = keyTasks[i].tensor.ReadbackAndClone() as Tensor<float>;
//                 //     pastValues[i] = valueTasks[i].tensor.ReadbackAndClone() as Tensor<float>;
//                 // }
//             for (int i = 0; i < MAX_LAYERS; i++)
//             {
//                 Debug.Log($"Qwen:11:{step}:{i}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//                 // Check if the output is ready
//                 // while (!_engine.IsOutputReady($"present.{i}.key"))
//                 // {
//                 //     yield return null;
//                 // }
//                 // Tensor output;
//                 yield return ReadOutputCoroutine(_engine, $"present.{i}.key", (tensor) => pastKeys[i] = tensor as Tensor<float>);

//                 // pastKeys[i] = output as Tensor<float>;
//                 // pastKeys[i] = await _engine.PeekOutput($"present.{i}.key").ReadbackAndCloneAsync() as Tensor<float>;
//                 Debug.Log($"Qwen:12:{step}:{i}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//                 // pastValues[i] = await _engine.PeekOutput($"present.{i}.value").ReadbackAndCloneAsync() as Tensor<float>;
//                 yield return ReadOutputCoroutine(_engine, $"present.{i}.value", (tensor) => pastValues[i] = tensor as Tensor<float>);
//                 Debug.Log($"Qwen:13:{step}:{i}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//                 _engine.SetInput($"past_key_values.{i}.key", pastKeys[i]);
//                 _engine.SetInput($"past_key_values.{i}.value", pastValues[i]);
//             }

//             int currentSequenceLength = initialTokenCount + step;

//             Debug.Log($"Qwen:14:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");
//             using var newInputTensor = new Tensor<int>(new TensorShape(1, 1), new[] { nextToken });
//             using var newPositionIdsTensor = new Tensor<int>(new TensorShape(1, 1), new[] { currentSequenceLength - 1 });
//             using var newAttentionMaskTensor = new Tensor<int>(new TensorShape(1, currentSequenceLength), Enumerable.Repeat(1, currentSequenceLength).ToArray());
//             Debug.Log($"Qwen:15:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//             _engine.SetInput("input_ids", newInputTensor);
//             _engine.SetInput("attention_mask", newAttentionMaskTensor);
//             _engine.SetInput("position_ids", newPositionIdsTensor);

//             Debug.Log($"Qwen:16:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//             iterator = _engine.ScheduleIterable();

//             iterationCount = 0;
//             while (iterator.MoveNext())
//             {
//                 iterationCount++;

//                 // Yield every ENGINE_ITERATIONS iterations to keep Unity responsive
//                 if (iterationCount % ENGINE_ITERATIONS == 0)
//                 {
//                     yield return null;
//                 }
//             }
//             Debug.Log($"Qwen:17:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//             // using var newOutputLogits = _engine.PeekOutput("logits") as Tensor<float>;
//             Tensor<float> newOutputLogits = null;
//             yield return ReadOutputCoroutine(_engine, $"logits", (tensor) => newOutputLogits = tensor as Tensor<float>);

//             Debug.Log($"Qwen:18:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//             // nextToken = await ProcessLogits(newOutputLogits, 0);
//             yield return ProcessLogits(newOutputLogits, 0, (token) => nextToken = token);

//             Debug.Log($"Qwen:19:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//             if (nextToken != TOKEN_IM_END)
//             {
//                 _tokens.Add(nextToken);
//                 _outputTokens.Add(nextToken);
//             }
//             Debug.Log($"Qwen:20:{step}:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//             // textLabel.text = _tokenizer.Decode(_outputTokens);
//             //Debug.Log($"Step {step}: Generated token: {nextToken} - {_tokenizer.Decode(new List<int> { nextToken })}");

//             callback(_tokenizer.Decode(_outputTokens));
//             step++;
//         }

//         Debug.Log($"Qwen:21:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         for (int i = 0; i < MAX_LAYERS; i++)
//         {
//             pastKeys[i]?.Dispose();
//             pastValues[i]?.Dispose();
//         }
//         Debug.Log($"Qwen:22:Elapsed time: {stopwatch.ElapsedMilliseconds} ms");

//         string generatedText = _tokenizer.Decode(_outputTokens);
//         Debug.Log($"Final sequence: {generatedText}");

//         stopwatch.Stop();
//         Debug.Log($"<color=cyan><b>Total Generation Time: {stopwatch.ElapsedMilliseconds} ms</b></color>");

//         // return generatedText;
//         callback(generatedText);
//     }

//     private IEnumerator ReadOutputCoroutine(Worker engine, string outputName, Action<Tensor> callback)
//     {
//         Tensor buffer = engine.PeekOutput(outputName);

//         if (buffer?.dataOnBackend == null)
//         {
//             callback(null);
//             yield break;
//         }

//         // Request readback
//         buffer.ReadbackRequest();

//         // Wait for readback to complete
//         while (!buffer.IsReadbackRequestDone())
//         {
//             yield return null;
//         }

//         // Return the cloned tensor via callback
//         callback(buffer.ReadbackAndClone());
//     }

//     private async Task<int> ProcessLogitsOld(Tensor<float> logits, int sequenceIndex)
//     {
//         _decodeEngine.SetInput(0, logits);
//         _decodeEngine.Schedule();
//         using var argMaxTensor = await _decodeEngine.PeekOutput().ReadbackAndCloneAsync() as Tensor<int>;
//         int nextToken = argMaxTensor.DownloadToArray()[sequenceIndex];
//         //string tokenStr = _tokenizer.Decode(new List<int> { nextToken });
//         return nextToken;
//     }

//     private IEnumerator ProcessLogits(Tensor<float> logits, int sequenceIndex, Action<int> callback)
//     {
//         _decodeEngine.SetInput(0, logits);
//         _decodeEngine.Schedule();

//         IEnumerator iterator = _decodeEngine.ScheduleIterable();

//         int iterationCount = 0;
//         while (iterator.MoveNext())
//         {
//             iterationCount++;

//             // Yield every ENGINE_ITERATIONS iterations to keep Unity responsive
//             if (iterationCount % ENGINE_ITERATIONS == 0)
//             {
//                 yield return null;
//             }
//         }
//         Tensor<int> output = null;
//         yield return ReadOutputCoroutine(_decodeEngine, "output_0", (tensor) => output = tensor as Tensor<int>);

//         // using var argMaxTensor = await _decodeEngine.PeekOutput().ReadbackAndCloneAsync() as Tensor<int>;
//         // int nextToken = argMaxTensor.DownloadToArray()[sequenceIndex];
//         int nextToken = output.DownloadToArray()[sequenceIndex];
//         //string tokenStr = _tokenizer.Decode(new List<int> { nextToken });
//         // return nextToken;
//         callback(nextToken);
//     }

//     private void OnDestroy()
//     {
//         _engine?.Dispose();
//         _decodeEngine?.Dispose();
//     }
// }
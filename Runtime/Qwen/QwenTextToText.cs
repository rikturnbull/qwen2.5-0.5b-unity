using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using XrAiAccelerator;

[XrAiProvider("Qwen")]
public class QwenTextToText : IXrAiTextToText
{
    private ModelQwen _modelQwen;
    
    public async Task Initialize(Dictionary<string, string> options = null, XrAiAssets assets = null)
    {
        _modelQwen = new ModelQwen();
        await _modelQwen.Start();        }

    public async Task Execute(string inputText, Dictionary<string, string> options, Action<XrAiResult<string>> callback)
    {
        await _modelQwen.Generate(inputText, (result) =>
        {
            callback(XrAiResult.Success(result));
        });
    }
}

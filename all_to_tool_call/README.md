## 大模型api为什么需要具备工具调用能力？
大多数的agent框架或者mcp客户端需要模型api支持工具调用能力

## 直接工具调用和间接工具调用
- 直接工具调用
  
api返回的结果中有字段专门存储工具调用的结果
- 间接工具调用
  
在提示词中让模型按照给定格式返回工具调用的结果，然后从content中解析出工具参数和名称
```
ChatCompletion(id='chatcmpl-8df49377-70bb-912c-a166-e35c222cb174', choices=[Choice(finish_reason='tool_calls', index=0, logprobs=None, message=ChatCompletionMessage(content='', refusal=None, role='assistant', annotations=None, audio=None, function_call=None, tool_calls=[ChatCompletionMessageFunctionToolCall(id='call_496e019007ec450285dbde', function=Function(arguments='{"location": "北京市"}', name='get_current_weather'), type='function', index=0)]))], created=1777838091, model='qwen2.5-7b-instruct', object='chat.completion', service_tier=None, system_fingerprint=None, usage=CompletionUsage(completion_tokens=18, prompt_tokens=276, total_tokens=294, completion_tokens_details=None, prompt_tokens_details=None))
```

## 实现流程
```mermaid
graph TD
    A[用户请求] --> B{大模型推理}
    B --> C[生成调用决策]
    C --> D{小模型解析}
    D --> E[执行工具调用]
    E --> F[返回结构化结果]
```
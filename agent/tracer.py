from pydantic_ai.messages import SystemPromptPart, ToolCallPart, TextPart, UserPromptPart, ToolReturnPart

def convert_to_json(message):
    if isinstance(message, SystemPromptPart):
        return {"system_prompt":{"role": "system", "content": message.content}}
    elif isinstance(message, ToolCallPart):
        return {"tool_call":{
            "role": "tool",
            "name": message.tool_name,
            "arguments": message.args,
        }}
    elif isinstance(message, TextPart):
        return {"text":{"role": "user", "content": message.content}}
    
    elif isinstance(message, UserPromptPart):
        return {"user_input":{"role": "user", "content": message.content}}
    elif isinstance(message, ToolReturnPart):
        return {"tool_return":{
            "role": "tool",
            "name": message.tool_name,
            "content": message.content,
        }}
    else:
        raise ValueError(f"Unknown message type: {type(message)}")\
            
def covert_to_trace(messages):
    spans = []
    for i, model_request in enumerate(messages):
        span = {"span_id": i, "messages": []}
        for part in model_request.parts:
            span['messages'].append(convert_to_json(part))
        spans.append(span)
    
    return spans
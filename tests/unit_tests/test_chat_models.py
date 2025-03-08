from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_openai.chat_models.base import (
    _convert_dict_to_message,
    _convert_delta_to_message_chunk,
)

def test_convert_dict_to_message_with_reasoning():
    """Test that reasoning is properly extracted from assistant messages."""
    # Test message with reasoning
    message_dict = {
        "role": "assistant",
        "content": "This is the content",
        "reasoning": "This is the reasoning"
    }
    message = _convert_dict_to_message(message_dict)
    
    assert isinstance(message, AIMessage)
    assert message.content == "This is the content"
    assert message.additional_kwargs["reasoning"] == "This is the reasoning"

    # Test message without reasoning
    message_dict = {
        "role": "assistant",
        "content": "This is the content"
    }
    message = _convert_dict_to_message(message_dict)
    
    assert isinstance(message, AIMessage)
    assert message.content == "This is the content"
    assert "reasoning" not in message.additional_kwargs

def test_convert_delta_to_message_chunk_with_reasoning():
    """Test that reasoning is properly extracted from delta chunks."""
    # Test chunk with reasoning
    delta_dict = {
        "role": "assistant",
        "content": "This is the content",
        "reasoning": "This is the reasoning"
    }
    chunk = _convert_delta_to_message_chunk(delta_dict, AIMessageChunk)
    
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.content == "This is the content"
    assert chunk.additional_kwargs["reasoning"] == "This is the reasoning"

    # Test chunk without reasoning
    delta_dict = {
        "role": "assistant",
        "content": "This is the content"
    }
    chunk = _convert_delta_to_message_chunk(delta_dict, AIMessageChunk)
    
    assert isinstance(chunk, AIMessageChunk)
    assert chunk.content == "This is the content"
    assert "reasoning" not in chunk.additional_kwargs
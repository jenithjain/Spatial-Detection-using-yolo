from django import template
import json

register = template.Library()

@register.filter
def pprint(value):
    """Parse a JSON string and return it as a dictionary or list."""
    if not value:
        return {}
    
    try:
        # Try to parse the JSON
        parsed_data = json.loads(value)
        
        # Check if it's a dictionary with key-value pairs
        if isinstance(parsed_data, dict):
            return parsed_data
        
        # Handle list format
        elif isinstance(parsed_data, list):
            # If it's just a list of strings, return directly
            if all(isinstance(item, str) for item in parsed_data):
                return parsed_data
                
            # Try to convert list to dictionary if possible
            result = {}
            for item in parsed_data:
                if isinstance(item, dict) and 'key' in item and 'value' in item:
                    result[item['key']] = item['value']
                elif isinstance(item, dict) and len(item) == 1:
                    key, value = next(iter(item.items()))
                    result[key] = value
            
            # Return the converted dictionary if we have values, otherwise return the original list
            return result if result else parsed_data
        
        return parsed_data
    except (ValueError, TypeError) as e:
        print(f"Error parsing JSON in pprint filter: {str(e)} - Value: {value}")
        return {}

@register.filter
def get_item(dictionary, key):
    """Get an item from a dictionary using its key."""
    if dictionary is None:
        return 0
    # Check if dictionary is actually a dictionary
    if not isinstance(dictionary, dict):
        return dictionary  # Return the value itself if it's not a dictionary
    return dictionary.get(key, 0)

@register.filter
def subtract(value, arg):
    """Subtract the arg from the value."""
    try:
        return int(value) - int(arg)
    except (ValueError, TypeError):
        return 0 
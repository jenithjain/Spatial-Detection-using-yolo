from django import template

register = template.Library()

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
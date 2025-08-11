"""Slack notification tools for Pipecat bot."""
import json
import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import aiohttp
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams

# Slack configuration
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_CHANNEL = "#test-hackathon-fruits"  # Default channel, can be overridden in function calls
SLACK_API_URL = "https://slack.com/api/chat.postMessage"

def load_live_tracking_data():
    """Load the live tracking data from JSON file."""
    try:
        with open("live_tracking_simple.json", "r") as f:
            data = json.load(f)
            return data.get("counts", {})
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

async def send_slack_message(message: str, channel: str = SLACK_CHANNEL, blocks: Optional[list] = None) -> Dict[str, Any]:
    """Send a message to Slack channel."""
    if not SLACK_BOT_TOKEN:
        raise Exception("SLACK_BOT_TOKEN environment variable not set")
    
    headers = {
        "Authorization": f"Bearer {SLACK_BOT_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "channel": channel,
        "text": message
    }
    
    # Add blocks for richer formatting if provided
    if blocks:
        payload["blocks"] = blocks
    
    async with aiohttp.ClientSession() as session:
        async with session.post(SLACK_API_URL, headers=headers, json=payload) as response:
            result = await response.json()
            if not result.get("ok"):
                raise Exception(f"Slack API error: {result.get('error', 'Unknown error')}")
            return result

def format_fruit_message(fruits: Dict[str, int]) -> tuple[str, list]:
    """Format fruit inventory into a Slack message with blocks."""
    # Plain text version
    text = "🍎 Fruit Inventory Update 🍎\n"
    for fruit, count in fruits.items():
        text += f"• {fruit}: {count}\n"
    
    # Rich blocks version
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🍎 Fruit Inventory Update 🍎",
                "emoji": True
            }
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*{fruit.capitalize()}:* {count}"
                }
                for fruit, count in fruits.items()
            ]
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_Updated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_"
                }
            ]
        }
    ]
    
    return text, blocks

# Define the Slack notification function schema
slack_notify_schema = FunctionSchema(
    name="send_fruit_update",
    description="Send fruit inventory update to Slack channel",
    properties={
        "channel": {
            "type": "string",
            "description": "Slack channel to send to (optional, defaults to #general)"
        },
        "custom_message": {
            "type": "string",
            "description": "Custom message to send instead of automatic fruit counts"
        },
        "include_all_fruits": {
            "type": "boolean",
            "description": "Whether to include all tracked fruits in the update"
        }
    },
    required=[]
)

# Define the check and notify schema (combines checking counts and sending to Slack)
check_and_notify_schema = FunctionSchema(
    name="check_and_notify_fruits",
    description="Check current fruit counts and send update to Slack",
    properties={
        "fruit_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of fruit types to check and report, or empty for all"
        },
        "channel": {
            "type": "string",
            "description": "Slack channel to send to (optional)"
        },
        "alert_if_low": {
            "type": "boolean",
            "description": "Send alert if any fruit count is below 10"
        }
    },
    required=[]
)

# Your existing object tracking schemas
object_tracking_schema = FunctionSchema(
    name="check_object_counts",
    description="Check the current counts of detected objects from live tracking",
    properties={
        "object_type": {
            "type": "string",
            "description": "Type of object to check or 'all' for all detected objects"
        }
    },
    required=["object_type"]
)

update_object_schema = FunctionSchema(
    name="update_object_count",
    description="Update the count of a specific object in the tracking system",
    properties={
        "object_type": {
            "type": "string",
            "description": "Type of object to update"
        },
        "quantity": {
            "type": "number",
            "description": "New quantity for the object"
        }
    },
    required=["object_type", "quantity"]
)

# Create tools schema with all functions
tools = ToolsSchema(standard_tools=[
    object_tracking_schema, 
    update_object_schema,
    slack_notify_schema,
    check_and_notify_schema
])

# Function handler for sending Slack updates
async def handle_fruit_update(params: FunctionCallParams):
    """Handle sending fruit inventory updates to Slack."""
    channel = params.arguments.get("channel", SLACK_CHANNEL)
    custom_message = params.arguments.get("custom_message")
    include_all = params.arguments.get("include_all_fruits", False)
    
    try:
        if custom_message:
            # Send custom message
            result = await send_slack_message(custom_message, channel)
            await params.result_callback({
                "success": True,
                "message": "Custom message sent to Slack",
                "channel": channel
            })
        elif include_all:
            # Send all fruit counts
            fruits = load_live_tracking_data()
            if not fruits:
                await params.result_callback({
                    "error": "No fruit data available"
                })
                return
            
            text, blocks = format_fruit_message(fruits)
            result = await send_slack_message(text, channel, blocks)
            
            await params.result_callback({
                "success": True,
                "message": "Fruit inventory sent to Slack",
                "channel": channel,
                "fruits_reported": list(fruits.keys())
            })
        else:
            await params.result_callback({
                "error": "Please specify either custom_message or set include_all_fruits to true"
            })
    except Exception as e:
        await params.result_callback({
            "error": f"Failed to send Slack message: {str(e)}"
        })

# Handler for check and notify
async def handle_check_and_notify(params: FunctionCallParams):
    """Check fruit counts and send update to Slack."""
    fruit_types = params.arguments.get("fruit_types", [])
    channel = params.arguments.get("channel", SLACK_CHANNEL)
    alert_if_low = params.arguments.get("alert_if_low", False)
    
    try:
        all_fruits = load_live_tracking_data()
        
        # Filter fruits if specific types requested
        if fruit_types:
            fruits_to_report = {
                fruit: count for fruit, count in all_fruits.items() 
                if fruit in fruit_types
            }
        else:
            fruits_to_report = all_fruits
        
        if not fruits_to_report:
            await params.result_callback({
                "error": "No matching fruits found in tracking data"
            })
            return
        
        # Check for low inventory
        low_fruits = {
            fruit: count for fruit, count in fruits_to_report.items() 
            if count < 10
        } if alert_if_low else {}
        
        # Create message
        if low_fruits:
            # Alert message for low inventory
            text = "⚠️ LOW INVENTORY ALERT ⚠️\n"
            for fruit, count in low_fruits.items():
                text += f"• {fruit}: Only {count} left!\n"
            
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "⚠️ Low Inventory Alert",
                        "emoji": True
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*The following fruits are running low:*"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*{fruit.capitalize()}:* `{count}` remaining"
                        }
                        for fruit, count in low_fruits.items()
                    ]
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": "*Full Inventory:*"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"{fruit.capitalize()}: {count}"
                        }
                        for fruit, count in fruits_to_report.items()
                    ]
                }
            ]
        else:
            # Regular inventory update
            text, blocks = format_fruit_message(fruits_to_report)
        
        result = await send_slack_message(text, channel, blocks)
        
        await params.result_callback({
            "success": True,
            "message": "Inventory update sent to Slack",
            "channel": channel,
            "fruits_reported": list(fruits_to_report.keys()),
            "low_inventory_alerts": list(low_fruits.keys()) if low_fruits else []
        })
        
    except Exception as e:
        await params.result_callback({
            "error": f"Failed to check and notify: {str(e)}"
        })

# Your existing handlers
async def handle_object_counts(params: FunctionCallParams):
    """Handle checking object counts."""
    object_type = params.arguments.get("object_type")
    object_counts = load_live_tracking_data()
    
    if object_type == "all":
        result = object_counts
    elif object_type in object_counts:
        result = {object_type: object_counts[object_type]}
    else:
        result = {"error": f"Object type '{object_type}' not found in tracking data"}
    
    await params.result_callback(result)

async def handle_update_object(params: FunctionCallParams):
    """Handle updating object counts."""
    object_type = params.arguments.get("object_type")
    quantity = params.arguments.get("quantity")
    
    # Load current data
    current_data = {}
    try:
        with open("live_tracking_simple.json", "r") as f:
            current_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        current_data = {"counts": {}}
    
    # Update the count
    old_quantity = current_data.get("counts", {}).get(object_type, 0)
    current_data.setdefault("counts", {})[object_type] = int(quantity)
    
    # Save back to file
    try:
        with open("live_tracking_simple.json", "w") as f:
            json.dump(current_data, f, indent=2)
        
        result = {
            "success": True,
            "object": object_type,
            "old_quantity": old_quantity,
            "new_quantity": int(quantity)
        }
    except Exception as e:
        result = {"error": f"Failed to update tracking data: {str(e)}"}
    
    await params.result_callback(result)

def register_all_functions(llm):
    """Register all functions including Slack notifications with the LLM service."""
    # Register existing object tracking functions
    llm.register_function("check_object_counts", handle_object_counts)
    llm.register_function("update_object_count", handle_update_object)
    
    # Register new Slack functions
    llm.register_function("send_fruit_update", handle_fruit_update)
    llm.register_function("check_and_notify_fruits", handle_check_and_notify)

# For backwards compatibility
register_object_tracking_functions = register_all_functions

# Example standalone usage (for testing)
async def test_slack_integration():
    """Test function to demonstrate Slack integration."""
    # Create mock params for testing
    class MockParams:
        def __init__(self):
            self.arguments = {
                "include_all_fruits": True,
                "channel": "#general"
            }
        
        async def result_callback(self, result):
            print("Result:", result)
    
    # Test sending fruit update
    print("Testing Slack integration...")
    await handle_fruit_update(MockParams())

if __name__ == "__main__":
    # Run test
    asyncio.run(test_slack_integration())
"""Fruit inventory function tools for Pipecat bot."""

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams


# Inventory data
fruit_inventory = {
    "apples": 15,
    "bananas": 8,
    "pears": 12
}


# Define the fruit inventory function schema
fruit_inventory_schema = FunctionSchema(
    name="check_fruit_inventory",
    description="Check the current inventory of fruits in stock",
    properties={
        "fruit_type": {
            "type": "string",
            "enum": ["apples", "bananas", "pears", "all"],
            "description": "Type of fruit to check or 'all' for entire inventory"
        }
    },
    required=["fruit_type"]
)


# Define the update inventory function schema
update_inventory_schema = FunctionSchema(
    name="update_fruit_inventory",
    description="Update the quantity of a specific fruit in inventory",
    properties={
        "fruit_type": {
            "type": "string",
            "enum": ["apples", "bananas", "pears"],
            "description": "Type of fruit to update"
        },
        "quantity": {
            "type": "number",
            "description": "New quantity for the fruit"
        }
    },
    required=["fruit_type", "quantity"]
)


# Create tools schema
tools = ToolsSchema(standard_tools=[fruit_inventory_schema, update_inventory_schema])


# Function handler for checking inventory
async def handle_fruit_inventory(params: FunctionCallParams):
    fruit_type = params.arguments.get("fruit_type")
    
    if fruit_type == "all":
        result = fruit_inventory
    elif fruit_type in fruit_inventory:
        result = {fruit_type: fruit_inventory[fruit_type]}
    else:
        result = {"error": f"Unknown fruit type: {fruit_type}"}
    
    await params.result_callback(result)


# Handler for updating inventory
async def handle_update_inventory(params: FunctionCallParams):
    fruit_type = params.arguments.get("fruit_type")
    quantity = params.arguments.get("quantity")
    
    if fruit_type in fruit_inventory:
        old_quantity = fruit_inventory[fruit_type]
        fruit_inventory[fruit_type] = int(quantity)
        result = {
            "success": True,
            "fruit": fruit_type,
            "old_quantity": old_quantity,
            "new_quantity": int(quantity)
        }
    else:
        result = {"error": f"Unknown fruit type: {fruit_type}"}
    
    await params.result_callback(result)


def register_fruit_functions(llm):
    """Register all fruit inventory functions with the LLM service."""
    llm.register_function("check_fruit_inventory", handle_fruit_inventory)
    llm.register_function("update_fruit_inventory", handle_update_inventory)
import json

from groq import Groq
from django.conf import settings

from .tools import (
    get_order_details,
    get_refund_history,
    check_delivery_status,
)
from .models import Conversation, Message, AgentLog


# Initialize Groq client
client = Groq(
    api_key=settings.GROQ_API_KEY
)

groq_model = settings.GROQ_MODEL


# SUPPORT SYSTEM PROMPT --> Maya's job description
SUPPORT_SYSTEM_PROMPT = """
You are Maya, a customer support agent at CoolBreeze AC.
You help customers with issues related to their AC orders.

Your responsibilities:
- Always use your tools to gather facts before responding
- Check order details when customer mentions their order
- Check refund history before making any refund decisions
- Be empathetic but honest

Your personality:
- Friendly and professional
- Patient even when customer is angry
- Clear and concise in your replies
- No emojis

Important rules:
- Do not give any outside information which is not related to order or refund or delivery status
- Always check order details first before responding
- Never approve or deny a refund yourself
- If refund decision is needed — tell customer you are checking with your team
"""


MANAGER_SYSTEM_PROMPT = """
You are a senior support manager at CoolBreeze AC.
A support agent has escalated a customer case to you for a refund decision.

Your responsibilities:
- Review the case summary carefully
- Consider the customer's refund history
- Make a fair and final refund decision
- Give a clear reason for your decision

Your decision options:
- Approve refund — if the case is genuine and within policy
- Deny refund — if the case is suspicious or outside policy
- Escalate to risk team — if you suspect fraud

Important rules:
- Be fair but firm
- Base decision on facts — not emotions
- Always give a specific reason for your decision
- Keep your response concise and professional
"""

# SUPPORT TOOLS --> Groq/OpenAI-compatible tool schemas
SUPPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": (
                "Fetch complete order details including status, carrier, "
                "tracking number and days since order was placed. "
                "Use this when customer mentions their order or complains "
                "about delivery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The order ID to look up",
                    }
                },
                "required": ["order_id"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "get_refund_history",
            "description": (
                "Get complete refund history for a user. "
                "Use this before making any refund related decisions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": (
                            "The user ID to check refund history for"
                        ),
                    }
                },
                "required": ["user_id"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "check_delivery_status",
            "description": (
                "Check current delivery status using tracking number "
                "and carrier. Use this when customer complains about "
                "delayed or missing delivery."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {
                        "type": "string",
                        "description": "The shipment tracking number",
                    },
                    "carrier": {
                        "type": "string",
                        "description": (
                            "The carrier name, for example "
                            "BlueDart or Delhivery"
                        ),
                    },
                },
                "required": ["tracking_number", "carrier"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "escalate_to_manager",
            "description": (
                "Escalate the case to the manager for a refund decision. "
                "Use this when customer requests a refund or compensation. "
                "Prepare a detailed case summary including order details, "
                "refund history and customer complaint before escalating."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "case_summary": {
                        "type": "string",
                        "description": (
                            "Complete case summary including "
                            "order details, refund history and "
                            "customer complaint"
                        ),
                    },
                },
                "required": ["case_summary"],
            },
        },
    },
]


# execute_tool() --> Bridge between Groq and Python functions
def execute_tool(tool_name, tool_input):

    if tool_name == "get_order_details":
        return get_order_details(
            tool_input["order_id"]
        )

    if tool_name == "get_refund_history":
        return get_refund_history(
            tool_input["user_id"]
        )

    if tool_name == "check_delivery_status":
        return check_delivery_status(
            tool_input["tracking_number"],
            tool_input["carrier"],
        )

    if tool_name == "escalate_to_manager":
        case_summary = tool_input["case_summary"]
        print("escalating to manager=====>", case_summary)
        decision = run_manager_agent(case_summary)
        print("decision===>", decision)
        return decision

    raise ValueError(
        f"Unknown tool: {tool_name}"
    )


# Agent Loop --> loops until the task is done
def run_support_agent(
    user_message,
    conversation_id,
    order_id,
    user_id
):

    conv = Conversation.objects.get(
        id=conversation_id
    )

    conversation_messages = []

    for msg in conv.messages.order_by("created_at"):
        conversation_messages.append({
            "role": msg.role,
            "content": msg.content,
        })

    while True:

        # Send conversation to Groq
        response = client.chat.completions.create(
            model=groq_model,
            max_tokens=1024,

            messages=[
                {
                    "role": "system",
                    "content": (
                        SUPPORT_SYSTEM_PROMPT
                        + f"\n\nContext: This conversation is about "
                          f"Order #{order_id}, user: {user_id}"
                    ),
                },
                *conversation_messages,
            ],

            tools=SUPPORT_TOOLS,
        )

        # Get assistant message
        message = response.choices[0].message

        # Check why model stopped
        finish_reason = response.choices[0].finish_reason

        # ------------------------------------------------
        # MODEL WANTS TO USE A TOOL
        # ------------------------------------------------

        if finish_reason == "tool_calls":

            # Add assistant message containing tool calls
            conversation_messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ],
            })

            # Execute each tool
            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name

                # Groq returns arguments as a JSON string
                tool_input = json.loads(
                    tool_call.function.arguments
                )

                # Execute Python function
                result = execute_tool(
                    tool_name,
                    tool_input
                )

                # Add tool result to conversation
                conversation_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                })

            # Send tool results back to Groq
            continue

        # ------------------------------------------------
        # MODEL HAS FINAL ANSWER
        # ------------------------------------------------

        return message.content

def run_manager_agent(case_summary):
    conversation_messages = [{
        'role': 'user',
        'content': case_summary
    }]

    while True:
        response = client.chat.completions.create(
            model=groq_model,
            max_tokens=1024,
            messages=[
                {
                    'role': 'system',
                    'content': MANAGER_SYSTEM_PROMPT,
                },
                *conversation_messages,
            ]
        )

        message = response.choices[0].message

        finish_reason = response.choices[0].finish_reason

        if finish_reason == 'tool_calls':
            conversation_messages.append({
                'role': 'assistant',
                'content': message.content or "",
                'tool_calls': [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            'name': tool_call.function.name,
                            'arguments': tool_call.function.arguments
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            })

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name

                tool_input = json.loads(tool_call.function.arguments)

                result = execute_tool(tool_name, tool_input)

                conversation_messages.append({
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'content': str(result)
                })

            continue

        return message.content
        
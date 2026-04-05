# Customer Support Triage OpenEnv Environment

A real-world OpenEnv environment for training and evaluating AI agents on customer support ticket triage and resolution tasks.

## Environment Description

The **Customer Support Triage** environment simulates the daily tasks of a customer support agent. Agents must handle incoming tickets by searching for information, replying to users, or escalating complex issues to the appropriate team.

### Motivation

AI agents are increasingly used in customer support. This environment provides a standardized way to evaluate their ability to:
1.  Understand user intent.
2.  Use tools (search, escalation) effectively.
3.  Provide helpful and accurate resolutions.

## Space Definitions

### Observation Space

The observation is a Pydantic model with the following fields:
- `ticket_id`: Unique identifier for the ticket.
- `content`: The current message or system output.
- `history`: A list of previous messages in the thread.
- `available_tools`: Tools the agent can use (`search`, `reply`, `escalate`).
- `metadata`: Additional context (e.g., user loyalty level).

### Action Space

The action is a Pydantic model with:
- `tool`: The tool to use (`search`, `reply`, or `escalate`).
- `args`: A dictionary of arguments for the tool.
  - `search`: `{"query": "..."}`
  - `reply`: `{"message": "..."}`
  - `escalate`: `{"reason": "..."}`

## Tasks

| Task ID | Name | Difficulty | Description |
| :--- | :--- | :--- | :--- |
| `easy-password-reset` | Easy Password Reset | Easy | Help a user reset their password. |
| `medium-billing-dispute` | Medium Billing Dispute | Medium | Resolve a billing dispute with missing info. |
| `hard-technical-issue` | Hard Technical Issue | Hard | Resolve a complex technical issue requiring multiple steps. |

## Reward Function

The environment provides a dense reward signal:
- **Success**: Up to +1.0 for correct resolution or escalation.
- **Partial Progress**: Small positive rewards for helpful intermediate steps (e.g., searching for relevant info).
- **Efficiency Penalty**: -0.01 per step to encourage faster resolutions.
- **Failure**: 0.0 for incorrect or unhelpful actions.

## Setup and Usage

### Prerequisites

- Python 3.10+
- Docker (optional)

### Installation

1.  Clone the repository.
2.  Install dependencies (using `uv` is recommended):
    ```bash
    uv pip install -r requirements.txt
    ```

### Running the Server

To test the environment locally via the API:
```bash
uv run server.py
# or
uvicorn server:app --reload --port 7860
```

### Running the Baseline

To run the baseline agent and see scores for all tasks:
```bash
python baseline.py
```

### Submission

To submit your environment to the OpenEnv repository:
1.  Ensure `openenv.yaml` is correctly configured.
2.  Push to Hugging Face:
    ```bash
    openenv push --repo-id your-username/customer-support-triage
    ```

### Baseline Scores

| Task ID | Baseline Score |
| :--- | :--- |
| `easy-password-reset` | 0.99 |
| `medium-billing-dispute` | 0.98 |
| `hard-technical-issue` | 0.98 |

## Deployment

### Hugging Face Spaces

This environment is designed to be deployed as a Docker-based Hugging Face Space.
1.  Create a new Space on Hugging Face.
2.  Select **Docker** as the SDK.
3.  Upload all files in this repository.

### Docker

To build and run locally:
```bash
docker build -t support-env .
docker run support-env
```

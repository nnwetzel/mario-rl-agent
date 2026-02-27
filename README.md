"""
README.md
Foundational Mario RL agent project.
"""

# Mario RL Agent

A foundational reinforcement learning agent to beat Super Mario Bros. World 1-1 using OpenAI Gym and gym-super-mario-bros.

## Project Structure
- `main.py`: Entry point, runs Mario environment with random actions.
- `agent.py`: Skeleton MarioAgent class.
- `train.py`: Basic training loop using MarioAgent.
- `requirements.txt`: Python dependencies.

## Getting Started
1. Create a virtual environment with Python 3.10 or 3.11 (recommended):
   ```
   python3.10 -m venv .venv
   # or
   python3.11 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the environment:
   ```
   python main.py
   ```
4. (TODO) Run the training loop:
   ```
   python train.py
   ```

## Next Steps
- Implement learning algorithms (DQN, PPO, etc.)
- Add reward shaping and evaluation
- Expand agent capabilities

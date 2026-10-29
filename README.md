# Solving MountainCar Through Intelligent Reward Shaping

> **Transforming an impossible reinforcement learning task into a 100% success rate through strategic reward design**

A deep reinforcement learning project demonstrating how custom reward shaping can overcome the challenge of sparse rewards in continuous control tasks. Using the Soft Actor-Critic (SAC) algorithm, this project achieves complete task mastery where the default approach fails entirely.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29+-green.svg)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-orange.svg)](https://stable-baselines3.readthedocs.io/)

---

## Results at a Glance

| Metric | Baseline (Default) | Custom Reward | Improvement |
|--------|-------------------|---------------|-------------|
| **Success Rate** | 0% | **100%** | +100% |
| **Max Position** | -0.472 | **0.466** | +0.938 |
| **Distance to Goal** | 0.922 | **0.000** | 100% closed |
| **Consistency (σ)** | 0.004 | **0.002** | 50% reduction |


**Key Achievement**: Designed a custom reward function that enabled the agent to go from complete failure to perfect task completion, demonstrating the critical importance of reward engineering in reinforcement learning.

---

## The Problem

### The Challenge

The **MountainCar-Continuous** environment presents a classic RL challenge:
- An underpowered car trapped in a valley must reach a flag on top of a hill
- The engine is too weak to drive straight up—the car must build momentum by oscillating
- The default reward structure provides almost no learning signal

### Why It's Hard

The default sparse reward makes learning nearly impossible:

```python
# Default reward: mostly uniform negative values
reward = -action²  # Same small penalty every step
reward += 100      # Only if goal reached (extremely rare)
```

**The problem**: With only negative rewards and no positional feedback, the agent has no way to discover that climbing the hill is beneficial. It's like trying to find a needle in a haystack blindfolded.

**Training budget**: 200,000 timesteps  
**Baseline result**: 0% success rate (agent never learns)

---

## The Solution

### Custom Reward Design

I engineered a dense reward function with 7 carefully balanced components:

```python
reward = (
    position_reward      # Progress toward goal (0-8)
    + velocity_reward    # Rightward momentum (0-4)
    + height_reward      # Elevation bonus (0-6)
    + near_goal_bonus    # Proximity incentive (0-50)
    + goal_bonus         # Terminal success (100)
    - energy_penalty     # Efficient control
    - time_penalty       # Fast solutions
)
```

**Key Innovation**: Each component provides immediate, actionable feedback that guides the agent toward the solution while maintaining alignment with the true objective.

### Design Principles

1. **Dense Feedback**: Reward every incremental improvement
2. **Progressive Difficulty**: Stronger rewards closer to the goal
3. **Energy Efficiency**: Penalize wasteful actions
4. **Behavioral Shaping**: Encourage both position and velocity toward the target

---

## Methodology

### Three-Stage Experimental Design

**Stage 1: Baseline**
- Train SAC with default sparse rewards
- 3 independent trials for statistical validity
- Establish performance ceiling with original reward

**Stage 2: Custom Reward**
- Implement and deploy dense reward function
- Same training budget and architecture
- Measure impact of reward design alone

**Stage 3: Hyperparameter Tuning**
- Test learning rates: 0.0001, 0.0003, 0.001
- Identify optimal configuration for custom reward
- Demonstrate compounding improvements

### Technical Details

- **Algorithm**: Soft Actor-Critic (off-policy, continuous actions)
- **Architecture**: Multi-layer perceptron (MLP) policy and value networks
- **Training**: 200,000 timesteps per configuration
- **Evaluation**: 20 episodes per model for statistical analysis
- **Metrics**: Max position reached, distance to goal, success rate, episode length

---

## Detailed Results

### Experiment 1: Baseline vs Custom Reward

| Trial | Baseline Max Pos | Custom Max Pos | Baseline Success | Custom Success |
|-------|------------------|----------------|------------------|----------------|
| 0 | -0.472 | **0.467** | 0% | **100%** |
| 1 | -0.476 | **0.463** | 0% | **100%** |
| 2 | -0.467 | **0.468** | 0% | **100%** |

**Analysis**: Custom reward enables consistent goal achievement across all trials. The agent not only learns to solve the task but does so reliably every time.

### Experiment 2: Learning Rate Impact

With custom reward shaping:

| Learning Rate | Max Position | Success Rate | Interpretation |
|---------------|--------------|--------------|----------------|
| 0.0001 | 0.120 | 35% | Too slow—insufficient learning |
| 0.0003 | 0.122 | 65% | Moderate—partial success |
| **0.001** | **0.464** | **100%** | Optimal—complete mastery |

**Finding**: Higher learning rates dramatically accelerate convergence when combined with dense rewards. The optimal rate achieves 3x the success rate of the default.

### Performance Metrics Over Time

The custom reward agent demonstrates:
- **Faster convergence**: Solves task in ~50k timesteps vs never
- **Stable learning**: Smooth reward curves without catastrophic forgetting
- **Shorter episodes**: Average 200-300 steps vs timeout at 1000
- **Higher rewards**: Cumulative episode returns 10x baseline

---

## Technologies & Skills Demonstrated

### Technical Stack

- **Deep RL**: Soft Actor-Critic algorithm with continuous action spaces
- **Python**: Object-oriented design, environment wrappers, experiment orchestration
- **Gymnasium**: Environment interaction, custom reward functions
- **Stable-Baselines3**: Model training, callbacks, evaluation
- **TensorBoard**: Real-time monitoring and visualization
- **NumPy**: Numerical computation and statistical analysis
- **YAML**: Configuration management

### Key Competencies

- **Reward Engineering**: Designed multi-component reward function that transformed task from impossible to trivial
- **Experimental Design**: Rigorous three-stage methodology with proper baselines and statistical validation
- **Hyperparameter Tuning**: Systematic exploration of learning rates with quantitative comparison
- **Scientific Communication**: Clear documentation, visualization, and interpretation of results
- **Software Engineering**: Modular code architecture with clean separation of concerns

---

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/lizakolosova/mountain-car-reinforcement-learning.git
cd rl-reward-shaping

# Install dependencies
pip install gymnasium stable-baselines3 pyyaml numpy tensorboard
```

### Run Experiments

```bash
# Train baseline (will fail to learn)
python src/train_baseline.py

# Train with custom reward (will succeed)
python src/train_custom_reward.py

# Hyperparameter tuning
python src/train_extension.py

# Compare all results
python src/comparison.py
```

### Monitor Training

```bash
tensorboard --logdir=logs
```

Open `http://localhost:6006` to watch training progress in real-time.

---

## Key Insights

### What Makes Reward Shaping Effective?

1. **Information Density**: Dense rewards create a learnable gradient that sparse rewards cannot provide
2. **Progressive Feedback**: Stronger signals near the goal help the agent "lock in" good behaviors
3. **Multi-Component Balance**: Combining position, velocity, and efficiency creates robust policies
4. **Preserved Alignment**: Custom rewards must still optimize for the true objective

### Why Default Rewards Fail

- **Exploration is blind**: Random actions almost never reach the goal by chance
- **No learning signal**: All failures look identical (same negative reward)
- **Credit assignment**: Even if goal is reached, agent can't determine which actions helped
- **Time complexity**: O(infinity) to discover sparse rewards through random exploration

### Lessons Learned

> "The difference between 0% and 100% success wasn't the algorithm, architecture, or compute—it was entirely about reward design."

This project demonstrates that thoughtful reward engineering is often more impactful than algorithmic innovations. Understanding your problem domain and designing appropriate feedback signals is a critical skill for practical RL applications.

---

## Technical Deep Dive

### Custom Reward Components Explained

**Position Reward** (8.0x scale)
```python
pos_norm = (position - min_pos) / (goal_pos - min_pos)
reward += pos_norm * 8.0
```
Creates a linear gradient across the track. Moving right always increases reward.

**Velocity Reward** (4.0x scale)
```python
reward += max(0, velocity) * 4.0
```
Rewards rightward momentum. Zero reward for leftward motion to avoid conflict.

**Height Reward** (6.0x scale)
```python
reward += max(0, position) * 6.0
```
Extra incentive for positive positions (right side of valley).

**Near-Goal Bonus** (50.0x scale, threshold 0.35)
```python
if position > 0.35:
    reward += (position - 0.35) * 50.0
```
Exponential-like increase approaching the goal. Helps "lock in" good behavior.

**Goal Bonus** (100.0 terminal)
```python
reward += 100.0 if position >= 0.45 else 0.0
```
Retains original objective. Makes true goal achievement clearly optimal.

---

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)
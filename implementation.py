# Import all required dependencies
import textarena as ta
import os
from typing import Any, Dict, List, Optional, Optional, Generator
import openai
import numpy as np
from dataclasses import dataclass
from collections import deque
import json
from datetime import datetime

# Check available openAI models
ta.basic_agents.openai.models.list()

# Define the Agent, Critique, and the ActorCritique memory blocks
@dataclass
class ObservationMemory:
    """Store observation data for each agent"""
    agent_id: int
    state: str
    resources: Dict[str, int]  # Track current resources
    values: Dict[str, float]   # Track resource values
    action: str
    global_state: str
    reward: float

@dataclass
class FeedbackMemory:
    """Store critic feedback for actor guidance"""
    agent_id: int
    feedback: str
    advantage: float
    value_estimate: float

class ActorCriticMemory:
    """Manages memory for both agents in actor-critic system"""
    def __init__(self, max_size: int = 1000):
        self.observations = {0: deque(maxlen=max_size), 1: deque(maxlen=max_size)}
        self.feedbacks = {0: deque(maxlen=max_size), 1: deque(maxlen=max_size)}

    def add_observation(self, memory: ObservationMemory):
        self.observations[memory.agent_id].append(memory)

    def add_feedback(self, memory: FeedbackMemory):
        self.feedbacks[memory.agent_id].append(memory)

    def get_recent_observations(self, agent_id: int, k: int = 5) -> List[ObservationMemory]:
        return list(self.observations[agent_id])[-k:]

    def get_recent_feedback(self, agent_id: int, k: int = 3) -> List[FeedbackMemory]:
        return list(self.feedbacks[agent_id])[-k:]

# Define the states tracker (Also known as the critic), replacing this part with PPO is considered a different RL approach
class StateAnalyzer:
    """Analyzes trading state and provides feedback to actors"""
    def __init__(self, model_name: str = "gpt-4"):
        self.model = model_name
        self.system_prompt = """Analyze the trading state and provide strategic feedback:
1. Evaluate resource values and holdings
2. Calculate potential trade advantages
3. Suggest optimal trading strategies
4. Consider opponent's trading patterns"""

    def analyze(self, global_state: str, memories: Dict[int, List[ObservationMemory]]) -> Dict[int, FeedbackMemory]:
        """For a particular step taken by an agent, critic Analyzes the memories of each agent as well as the environment, then generates the feedback for each agent"""
        feedbacks = {}
        try:
            for agent_id in [0, 1]:
                recent_obs = memories[agent_id]
                prompt = self._generate_trading_prompt(agent_id, global_state, recent_obs)

                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=512
                )

                feedback = response.choices[0].message.content
                advantage = self._calculate_trade_advantage(recent_obs)
                value_est = self._estimate_portfolio_value(recent_obs)

                feedbacks[agent_id] = FeedbackMemory(
                    agent_id=agent_id,
                    feedback=feedback,
                    advantage=advantage,
                    value_estimate=value_est
                )

        except Exception as e:
            print(f"Error in trade analysis: {e}")
            # Provide neutral feedback on error
            for agent_id in [0, 1]:
                feedbacks[agent_id] = FeedbackMemory(
                    agent_id=agent_id,
                    feedback="Error in analysis",
                    advantage=0.0,
                    value_estimate=0.0
                )

        return feedbacks

    def _generate_trading_prompt(self, agent_id: int, global_state: str, observations: List[ObservationMemory]) -> str:
        return f"""
Agent {agent_id} Trade Analysis:
Current Resources: {observations[-1].resources if observations else 'Unknown'}
Resource Values: {observations[-1].values if observations else 'Unknown'}
Recent Trades: {[obs.action for obs in observations]}
Trading History Results: {[obs.reward for obs in observations]}

Analyze trading position and suggest optimal strategy."""

    def _calculate_trade_advantage(self, observations: List[ObservationMemory]) -> float:
        if not observations:
            return 0.0
        latest = observations[-1]
        return sum(amount * latest.values.get(resource, 0)
                  for resource, amount in latest.resources.items())

    def _estimate_portfolio_value(self, observations: List[ObservationMemory]) -> float:
        if not observations:
            return 0.0
        return sum(observations[-1].reward for _ in observations[-3:]) / 3

# Define the actions selector that perform the actions by observing current state and feedbacks from critique
class ActionSelector:
    """Generates trading actions based on observations and feedback"""
    def __init__(self, agent_id: int, model_name: str = "gpt-4o-mini"):
        self.agent_id = agent_id
        self.model = model_name
        self.system_prompt = """Your goal is to maximize resource value through strategic trading:
1. Evaluate current resources and their values
2. Prioritize high-value resource acquisition
3. Respond to offers with [Accept] or [Deny]
4. Make strategic trade offers using [Offer: X -> Y] format
5. Adapt to opponent's trading patterns"""

    def select_action(self, observation: str, recent_feedback: List[FeedbackMemory]) -> str:
        try:
            feedback_text = self._format_trading_feedback(recent_feedback)

            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"Game Context:\n{observation}\n\nTrading Feedback:\n{feedback_text}"}
                ],
                max_tokens=512
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error selecting trade action: {e}")
            return "[Deny]"  # Safe default action

    def _format_trading_feedback(self, feedbacks: List[FeedbackMemory]) -> str:
        if not feedbacks:
            return "No previous trading feedback available."

        return f"""
Recent Trading Results:
{feedbacks[-1].feedback}

Suggestions:
- Focus on trades that increase your resource values
- Adapt to opponent's trading patterns
- Consider long-term value maximization"""

# Define the main game loop
class ActorCriticLoop:
    """Manages the trading game loop"""
    def __init__(self,
                 env_id: str = "Negotiation-v0",
                 actor_model1_name: str = "gpt-4o-mini",
                 actor_model2_name: str = "gpt-4o-mini",
                 critic_model_name: str = "gpt-4o-mini",
                 isHuman: bool = False):

        # Create the game environment with the wrappers
        self.env = ta.make(env_id)
        self.env = ta.wrappers.LLMObservationWrapper(env=self.env)

        # Initialize the memory component
        self.memory = ActorCriticMemory()

        # Initialize the critic
        self.critic = StateAnalyzer(model_name=critic_model_name)

        # Initialize the actors in a wrapper which they can use to perform actions with
        self.actors = {
            0: ActionSelector(agent_id=0, model_name=actor_model1_name),
            1: ActionSelector(agent_id=1, model_name=actor_model2_name)
        }

        # For storing the rewards for actions performed by the actors
        self.metrics = {
            "rewards": {0: [], 1: []},
            "advantages": {0: [], 1: []},
            "values": {0: [], 1: []}
        }

        # To save the history of all actions performed while the game is running
        self.transaction_history = []

        # To enter into the loop as a human and manipulate the flow
        self.isHuman = isHuman

    def _format_step_output(self, step_num: int, title: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format step output as JSON"""
        return {
            "step": step_num,
            "title": title,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

    def _update_memory(self, observations: Dict[int, str], current_id: int, action: str, rewards: Dict[int, int], feedbacks: Dict[int, FeedbackMemory]):
        """Helper method to update memory storage"""
        try:
            obs_dict = json.loads(observations[current_id]) if isinstance(observations[current_id], str) else {}
            resources = obs_dict.get("resources", {})
            values = obs_dict.get("values", {})
        except (json.JSONDecodeError, AttributeError):
            resources = {}
            values = {}

        self.memory.add_observation(
            ObservationMemory(
                agent_id=current_id,
                state=observations[current_id],
                resources=resources,
                values=values,
                action=action,
                global_state=str(self.env.state),
                reward=rewards.get(current_id, 0)
            )
        )

        for agent_id, feedback in feedbacks.items():
            self.memory.add_feedback(feedback)
            self.metrics["rewards"][agent_id].append(rewards.get(agent_id, 0))
            self.metrics["advantages"][agent_id].append(feedback.advantage)
            self.metrics["values"][agent_id].append(feedback.value_estimate)

    def _wait_for_human(self, step_output: Dict[str, Any]):
        """Allow human to analyze each step"""
        if self.isHuman:
            print("\nStep Output:")
            print(json.dumps(step_output, indent=2))
            input("\nPress Enter to continue...\n")

    def run_episode(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        step_outputs = []
        step_counter = 1

        # Step 1: Initialize Environment
        observations = self.env.reset(seed=seed)
        init_output = self._format_step_output(
            step_counter,
            "Environment Initialization",
            {
                "seed": seed,
                "initial_observations": observations,
                "environment_state": str(self.env.state)
            }
        )
        step_outputs.append(init_output)
        self._wait_for_human(init_output)

        done = False

        while not done:
            # Only the numerical label 0 or 1 to decide which agent should play
            current_id = self.env.state.get("current_player")

            # Step 2: Analyze Current State
            step_counter += 1
            state_analysis = self._format_step_output(
                step_counter,
                "State Analysis",
                {
                    "player_id": current_id,
                    "observation": observations[current_id],
                    "recent_feedback": [f.feedback for f in self.memory.get_recent_feedback(current_id)]
                }
            )
            step_outputs.append(state_analysis)
            self._wait_for_human(state_analysis)

            # Step 3: Action Generation based on the environment observation the agent currently has + critics feedback
            step_counter += 1
            action = self.actors[current_id].select_action(
                observations[current_id],
                self.memory.get_recent_feedback(current_id)
            )

            action_output = self._format_step_output(
                step_counter,
                "Action Generation",
                {
                    "player_id": current_id,
                    "action": action,
                }
            )
            step_outputs.append(action_output)
            self._wait_for_human(action_output)

            # Step 4: Environment Step
            step_counter += 1
            next_obs, rewards, truncated, terminated, info = self.env.step(
                current_id, action
            )

            step_result = self._format_step_output(
                step_counter,
                "Environment Step",
                {
                    "action_result": {
                        "observations": next_obs,
                        "rewards": rewards or {0: 0, 1: 0},
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info
                    }
                }
            )
            step_outputs.append(step_result)
            self._wait_for_human(step_result)

            # Step 5: Critic Analysis
            step_counter += 1
            try:
                feedbacks = self.critic.analyze(
                    str(self.env.state),
                    {id: self.memory.get_recent_observations(id) for id in [0, 1]}
                )

                critic_output = self._format_step_output(
                    step_counter,
                    "Critic Analysis",
                    {
                        "feedbacks": {
                            str(aid): {
                                "feedback": f.feedback,
                                "advantage": f.advantage,
                                "value_estimate": f.value_estimate
                            } for aid, f in feedbacks.items()
                        }
                    }
                )
            except Exception as e:
                critic_output = self._format_step_output(
                    step_counter,
                    "Critic Analysis Error",
                    {"error": str(e)}
                )

            step_outputs.append(critic_output)
            self._wait_for_human(critic_output)

            # Step 6: Memory Update
            step_counter += 1
            self._update_memory(observations, current_id, action, rewards or {0: 0, 1: 0}, feedbacks)
            memory_output = self._format_step_output(
                step_counter,
                "Memory Update",
                {
                    "transaction": {
                        "turn": len(self.transaction_history) + 1,
                        "player": current_id,
                        "action": action,
                        "rewards": rewards or {0: 0, 1: 0}
                    }
                }
            )
            step_outputs.append(memory_output)
            self._wait_for_human(memory_output)

            observations = next_obs
            done = terminated or truncated

        # Final Step: Episode Summary
        step_counter += 1
        summary_output = self._format_step_output(
            step_counter,
            "Episode Summary",
            {
                "final_rewards": rewards or {0: 0, 1: 0},
                "metrics": {
                    metric: {
                        str(aid): float(np.mean(values) if values else 0)
                        for aid, values in metric_data.items()
                    }
                    for metric, metric_data in self.metrics.items()
                },
                "reason": info.get("reason", "Unknown"),
                "total_steps": step_counter
            }
        )
        step_outputs.append(summary_output)
        self._wait_for_human(summary_output)

        return step_outputs

# **Implementation (Pure Agents only)**
# Define the main game loop (Agent only loop without the critic)
class ActorOnlyLoop:
    """Manages the trading game loop"""
    def __init__(self,
                 env_id: str = "Negotiation-v0",
                 actor_model1_name: str = "gpt-4o-mini",
                 actor_model2_name: str = "gpt-4o-mini",
                 isHuman: bool = False):

        # Create the game environment with the wrappers
        self.env = ta.make(env_id)
        self.env = ta.wrappers.LLMObservationWrapper(env=self.env)

        # Initialize the memory component
        self.memory = ActorCriticMemory()
        self.actors = {
            0: ActionSelector(agent_id=0, model_name=actor_model1_name),
            1: ActionSelector(agent_id=1, model_name=actor_model2_name)
        }

        self.metrics = {
            "rewards": {0: [], 1: []},
            "advantages": {0: [], 1: []},
            "values": {0: [], 1: []}
        }

        self.transaction_history = []
        self.isHuman = isHuman

    def _format_step_output(self, step_num: int, title: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format step output as JSON"""
        return {
            "step": step_num,
            "title": title,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

    def _update_memory(self, observations: Dict[int, str], current_id: int, action: str, rewards: Dict[int, int], feedbacks: Dict[int, FeedbackMemory]):
        """Helper method to update memory storage"""
        try:
            obs_dict = json.loads(observations[current_id]) if isinstance(observations[current_id], str) else {}
            resources = obs_dict.get("resources", {})
            values = obs_dict.get("values", {})
        except (json.JSONDecodeError, AttributeError):
            resources = {}
            values = {}

        self.memory.add_observation(
            ObservationMemory(
                agent_id=current_id,
                state=observations[current_id],
                resources=resources,
                values=values,
                action=action,
                global_state=str(self.env.state),
                reward=rewards.get(current_id, 0)
            )
        )

        for agent_id, feedback in feedbacks.items():
            self.memory.add_feedback(feedback)
            self.metrics["rewards"][agent_id].append(rewards.get(agent_id, 0))
            self.metrics["advantages"][agent_id].append(feedback.advantage)
            self.metrics["values"][agent_id].append(feedback.value_estimate)

    def _wait_for_human(self, step_output: Dict[str, Any]):
        """Allow human to analyze each step"""
        if self.isHuman:
            print("\nStep Output:")
            print(json.dumps(step_output, indent=2))
            input("\nPress Enter to continue...\n")

    def run_episode(self, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        step_outputs = []
        step_counter = 1

        # Step 1: Initialize Environment
        observations = self.env.reset(seed=seed)
        init_output = self._format_step_output(
            step_counter,
            "Environment Initialization",
            {
                "seed": seed,
                "initial_observations": observations,
                "environment_state": str(self.env.state)
            }
        )
        step_outputs.append(init_output)
        self._wait_for_human(init_output)

        done = False

        while not done:
            current_id = self.env.state.get("current_player")

            # Step 2: Analyze Current State
            step_counter += 1
            state_analysis = self._format_step_output(
                step_counter,
                "State Analysis",
                {
                    "player_id": current_id,
                    "observation": observations[current_id],
                    "recent_feedback": [f.feedback for f in self.memory.get_recent_feedback(current_id)]
                }
            )
            step_outputs.append(state_analysis)
            self._wait_for_human(state_analysis)

            # Step 3: Action Generation
            step_counter += 1
            action = self.actors[current_id].select_action(
                observations[current_id],
                self.memory.get_recent_feedback(current_id)
            )

            action_output = self._format_step_output(
                step_counter,
                "Action Generation",
                {
                    "player_id": current_id,
                    "action": action,
                }
            )
            step_outputs.append(action_output)
            self._wait_for_human(action_output)

            # Step 4: Environment Step
            step_counter += 1
            next_obs, rewards, truncated, terminated, info = self.env.step(
                current_id, action
            )

            step_result = self._format_step_output(
                step_counter,
                "Environment Step",
                {
                    "action_result": {
                        "observations": next_obs,
                        "rewards": rewards or {0: 0, 1: 0},
                        "terminated": terminated,
                        "truncated": truncated,
                        "info": info
                    }
                }
            )
            step_outputs.append(step_result)
            self._wait_for_human(step_result)

            # Step 5: Feedbacks from critiqe component, implementation removed
            feedbacks = {}

            # Step 6: Memory Update
            step_counter += 1
            self._update_memory(observations, current_id, action, rewards or {0: 0, 1: 0}, feedbacks)
            memory_output = self._format_step_output(
                step_counter,
                "Memory Update",
                {
                    "transaction": {
                        "turn": len(self.transaction_history) + 1,
                        "player": current_id,
                        "action": action,
                        "rewards": rewards or {0: 0, 1: 0}
                    }
                }
            )
            step_outputs.append(memory_output)
            self._wait_for_human(memory_output)

            observations = next_obs
            done = terminated or truncated

        # Final Step: Episode Summary
        step_counter += 1
        summary_output = self._format_step_output(
            step_counter,
            "Episode Summary",
            {
                "final_rewards": rewards or {0: 0, 1: 0},
                "metrics": {
                    metric: {
                        str(aid): float(np.mean(values) if values else 0)
                        for aid, values in metric_data.items()
                    }
                    for metric, metric_data in self.metrics.items()
                },
                "reason": info.get("reason", "Unknown"),
                "total_steps": step_counter
            }
        )
        step_outputs.append(summary_output)
        self._wait_for_human(summary_output)

        return step_outputs

class WebSocketActorCriticLoop:
    """Websocket-friendly version of ActorCriticLoop"""
    def __init__(self,
                 env_id: str = "Negotiation-v0",
                 actor_model1_name: str = "gpt-4o-mini",
                 actor_model2_name: str = "gpt-4o-mini",
                 critic_model_name: str = "gpt-4o-mini"):
        
        self.env = ta.make(env_id)
        self.env = ta.wrappers.LLMObservationWrapper(env=self.env)
        self.memory = ActorCriticMemory()
        self.critic = StateAnalyzer(model_name=critic_model_name)
        self.actors = {
            0: ActionSelector(agent_id=0, model_name=actor_model1_name),
            1: ActionSelector(agent_id=1, model_name=actor_model2_name)
        }
        self.metrics = {
            "rewards": {0: [], 1: []},
            "advantages": {0: [], 1: []},
            "values": {0: [], 1: []}
        }
        self.transaction_history = []

    def _format_message(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format message for websocket transmission"""
        return {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

    def _update_memory(self, observations: Dict[int, str], current_id: int, action: str, 
                      rewards: Dict[int, int], feedbacks: Dict[int, FeedbackMemory]):
        """Helper method to update memory storage"""
        try:
            obs_dict = json.loads(observations[current_id]) if isinstance(observations[current_id], str) else {}
            resources = obs_dict.get("resources", {})
            values = obs_dict.get("values", {})
        except (json.JSONDecodeError, AttributeError):
            resources = {}
            values = {}

        self.memory.add_observation(
            ObservationMemory(
                agent_id=current_id,
                state=observations[current_id],
                resources=resources,
                values=values,
                action=action,
                global_state=str(self.env.state),
                reward=rewards.get(current_id, 0)
            )
        )

        for agent_id, feedback in feedbacks.items():
            self.memory.add_feedback(feedback)
            self.metrics["rewards"][agent_id].append(rewards.get(agent_id, 0))
            self.metrics["advantages"][agent_id].append(feedback.advantage)
            self.metrics["values"][agent_id].append(feedback.value_estimate)
        pass

    def run_episode(self, seed: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Generator version of run_episode for websocket streaming"""
        # Initialize Environment
        observations = self.env.reset(seed=seed)
        yield self._format_message("init", {
            "seed": seed,
            "initial_observations": observations,
            "environment_state": str(self.env.state)
        })

        done = False
        while not done:
            current_id = self.env.state.get("current_player")

            # State Analysis
            yield self._format_message("state", {
                "player_id": current_id,
                "observation": observations[current_id]
            })

            # Action Generation
            action = self.actors[current_id].select_action(
                observations[current_id],
                self.memory.get_recent_feedback(current_id)
            )
            yield self._format_message("action", {
                "player_id": current_id,
                "action": action
            })

            # Environment Step
            next_obs, rewards, truncated, terminated, info = self.env.step(current_id, action)
            yield self._format_message("step", {
                "observations": next_obs,
                "rewards": rewards or {0: 0, 1: 0},
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            })

            # Critic Analysis
            try:
                feedbacks = self.critic.analyze(
                    str(self.env.state),
                    {id: self.memory.get_recent_observations(id) for id in [0, 1]}
                )
                yield self._format_message("critique", {
                    "feedbacks": {
                        str(aid): {
                            "feedback": f.feedback,
                            "advantage": f.advantage,
                            "value_estimate": f.value_estimate
                        } for aid, f in feedbacks.items()
                    }
                })
            except Exception as e:
                yield self._format_message("error", {"error": str(e)})

            # Memory Update
            self._update_memory(observations, current_id, action, rewards or {0: 0, 1: 0}, feedbacks)
            observations = next_obs
            done = terminated or truncated

        # Episode Summary
        yield self._format_message("summary", {
            "final_rewards": rewards or {0: 0, 1: 0},
            "metrics": {
                metric: {
                    str(aid): float(np.mean(values) if values else 0)
                    for aid, values in metric_data.items()
                }
                for metric, metric_data in self.metrics.items()
            },
            "reason": info.get("reason", "Unknown")
        })

class WebSocketActorOnlyLoop:
    """Websocket-friendly version of ActorOnlyLoop"""
    def __init__(self,
                 env_id: str = "Negotiation-v0",
                 actor_model1_name: str = "gpt-4o-mini",
                 actor_model2_name: str = "gpt-4o-mini"):
        
        self.env = ta.make(env_id)
        self.env = ta.wrappers.LLMObservationWrapper(env=self.env)
        self.memory = ActorCriticMemory()
        self.actors = {
            0: ActionSelector(agent_id=0, model_name=actor_model1_name),
            1: ActionSelector(agent_id=1, model_name=actor_model2_name)
        }
        self.metrics = {
            "rewards": {0: [], 1: []},
            "advantages": {0: [], 1: []},
            "values": {0: [], 1: []}
        }
        self.transaction_history = []

    def _format_message(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }

    def _update_memory(self, observations: Dict[int, str], current_id: int, action: str, 
                      rewards: Dict[int, int], feedbacks: Dict[int, FeedbackMemory]):
        """Helper method to update memory storage"""
        try:
            obs_dict = json.loads(observations[current_id]) if isinstance(observations[current_id], str) else {}
            resources = obs_dict.get("resources", {})
            values = obs_dict.get("values", {})
        except (json.JSONDecodeError, AttributeError):
            resources = {}
            values = {}

        self.memory.add_observation(
            ObservationMemory(
                agent_id=current_id,
                state=observations[current_id],
                resources=resources,
                values=values,
                action=action,
                global_state=str(self.env.state),
                reward=rewards.get(current_id, 0)
            )
        )

        for agent_id, feedback in feedbacks.items():
            self.memory.add_feedback(feedback)
            self.metrics["rewards"][agent_id].append(rewards.get(agent_id, 0))
            self.metrics["advantages"][agent_id].append(feedback.advantage)
            self.metrics["values"][agent_id].append(feedback.value_estimate)
        pass

    def run_episode(self, seed: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Generator version without critic component"""
        # Initialize Environment
        observations = self.env.reset(seed=seed)
        yield self._format_message("init", {
            "seed": seed,
            "initial_observations": observations,
            "environment_state": str(self.env.state)
        })

        done = False
        while not done:
            current_id = self.env.state.get("current_player")

            # State Analysis
            yield self._format_message("state", {
                "player_id": current_id,
                "observation": observations[current_id]
            })

            # Action Generation
            action = self.actors[current_id].select_action(
                observations[current_id],
                self.memory.get_recent_feedback(current_id)
            )
            yield self._format_message("action", {
                "player_id": current_id,
                "action": action
            })

            # Environment Step
            next_obs, rewards, truncated, terminated, info = self.env.step(current_id, action)
            yield self._format_message("step", {
                "observations": next_obs,
                "rewards": rewards or {0: 0, 1: 0},
                "terminated": terminated,
                "truncated": truncated,
                "info": info
            })

            # Memory Update (without critic feedback)
            self._update_memory(observations, current_id, action, rewards or {0: 0, 1: 0}, {})
            observations = next_obs
            done = terminated or truncated

        # Episode Summary
        yield self._format_message("summary", {
            "final_rewards": rewards or {0: 0, 1: 0},
            "metrics": {
                metric: {
                    str(aid): float(np.mean(values) if values else 0)
                    for aid, values in metric_data.items()
                }
                for metric, metric_data in self.metrics.items()
            },
            "reason": info.get("reason", "Unknown")
        })

# Run the program
if __name__ == "__main__":
  try:
      ac_loop = ActorCriticLoop(
          env_id="Negotiation-v0",
          actor_model1_name="gpt-4o-mini",
          actor_model2_name="gpt-3.5-turbo",
          critic_model_name="gpt-3.5-turbo", # Disable for pure agents implementation
          isHuman=True
      )

      step_outputs = ac_loop.run_episode(seed=1)

      for step_output in step_outputs:
        print(step_output)

  except Exception as e:
      print(f"Error running episode: {e}")

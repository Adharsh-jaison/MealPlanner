from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
import os
from google.adk.tools import google_search
from google.adk.tools.exit_loop_tool import exit_loop


def save_meal_preferences(preferences: str):
    """
    Saves the collected meal preferences to preference.txt.
    Args:
        preferences: A string containing the user's meal preferences.
    """
    print(f"\n===== Saving Preferences: {preferences} =====")
    file_path = os.path.join(os.path.dirname(__file__), "preference.txt")
    with open(file_path, "w") as f:
        f.write(preferences)
    print(f"Preferences saved to {file_path}.")

def read_preference_file():
    """
    Reads the content of preference.txt.
    """
    file_path = os.path.join(os.path.dirname(__file__), "preference.txt")
    print(f"DEBUG: Attempting to read from: {file_path}")
    try:
        with open(file_path, "r") as f:
            content = f.read()
            print(f"DEBUG: File content length: {len(content)}")
            return content
    except FileNotFoundError:
        print("DEBUG: File not found.")
        return "No preferences found in preference.txt."


immune_agent = Agent(
    name="immune_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
Generate a 2-day meal plan for immune health.
Create a balanced meal plan with breakfast, lunch, and dinner for Day 1 and Day 2.
Focus on immune-boosting foods (citrus fruits, leafy greens, nuts, yogurt, etc.).
Use only 200 words.

IMPORTANT:
- DO NOT apologize.
- DO NOT ask for clarification.
- DO NOT chat.
- JUST GENERATE THE MEAL PLAN.
""",
    tools=[google_search],
    output_key="immune_plan"
)


metabolic_health_agent = Agent(
    name="metabolic_health_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
Generate a 2-day meal plan for metabolic health.
Create a balanced meal plan with breakfast, lunch, and dinner for Day 1 and Day 2.
Focus on metabolic health (whole grains, lean proteins, healthy fats, low glycemic foods).
Use only 200 words.

IMPORTANT:
- DO NOT apologize.
- DO NOT ask for clarification.
- DO NOT chat.
- JUST GENERATE THE MEAL PLAN.
""",
    tools=[google_search],
    output_key="metabolic_plan"
)


muscle_building_agent = Agent(
    name="muscle_building_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
Generate a 2-day meal plan for muscle building.
Create a balanced meal plan with breakfast, lunch, and dinner for Day 1 and Day 2.
Focus on muscle building (high protein, complex carbs, healthy fats).
Use only 200 words.

IMPORTANT:
- DO NOT apologize.
- DO NOT ask for clarification.
- DO NOT chat.
- JUST GENERATE THE MEAL PLAN.
""",
    tools=[google_search],
    output_key="muscle_plan"
)


condition_based_agent = Agent(
    name="condition_based_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
Generate a 2-day meal plan for condition-based dietary needs.
Create a balanced meal plan with breakfast, lunch, and dinner for Day 1 and Day 2.
Focus on general health conditions (diabetes-friendly, heart-healthy, anti-inflammatory).
Use only 200 words.

IMPORTANT:
- DO NOT apologize.
- DO NOT ask for clarification.
- DO NOT chat.
- JUST GENERATE THE MEAL PLAN.
""",
    tools=[google_search],
    output_key="condition_plan"
)


meal_plan_aggregator_agent = Agent(
    name="MealPlanAggregatorAgent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
    ),
    instruction="""
Combine the outputs from the four specialized meal plan agents into a single, coherent meal plan summary:

**Immune-Boosting Plan:**
{immune_plan}

**Metabolic Health Plan:**
{metabolic_plan}

**Muscle-Building / Weight Gain Plan:**
{muscle_plan}

**Condition-Based Plan:**
{condition_plan}

Your combined summary should highlight the key features of each plan.
Present the final 7-day meal plan in a clear Markdown table with the following columns:
| Day | Breakfast | Lunch | Dinner | Snacks |
|---|---|---|---|---|
| Day 1 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

Make it clear, professional, and encouraging.
""",
    output_key="combined_meal_plan"
)

meal_plan_validator_agent = Agent(
    name="meal_plan_validator_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a meal plan quality validator with a LENIENT approach.

Your job: Verify that the {combined_meal_plan} meets the MAJOR criteria from the {reviewed_preferences}.

Validation checklist (focus on major criteria):
1. Critical dietary restrictions (allergies, vegetarian/vegan if specified) - MUST be respected
2. Primary health goals (immune health, muscle building, etc.) - Should be addressed
3. Cuisine preferences - Should be incorporated where reasonable
4. Meal variety and balance - Should be present

Decision-making (LENIENT):
- If ALL critical dietary restrictions are respected AND primary health goals are addressed: Call `exit_loop` to proceed
- Only reject if there are MAJOR violations (e.g., meat in vegetarian plan, allergens present, completely ignores health goals)
- Minor issues (e.g., not every meal is Kerala cuisine, some variety in approach) are ACCEPTABLE

When rejecting:
- Clearly state what MAJOR criteria were violated
- Be specific about what needs to be fixed
- DO NOT call exit_loop (this triggers regeneration)

When accepting:
- Call `exit_loop` to proceed to post-processing
- Briefly confirm that major criteria are met

Be fair and lenient - don't reject for minor imperfections.
""",
    tools=[exit_loop],
    output_key="validation_result"
)


parallel_Meal_plan_agent = ParallelAgent(
    name="ParallelMealPlanAgent",
    sub_agents=[immune_agent, metabolic_health_agent, muscle_building_agent, condition_based_agent],
)


nutrition_agent = Agent(
    name="nutrition_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are an expert nutritionist.
Your goal is to analyze the provided meal plan and calculate the total nutritional value.

Use the "{combined_meal_plan}" which contains the 7-day meal plan and make a nutrition analysis based on the ingredients.
Present the analysis in a Markdown table with the following columns:
| Nutrient | Amount per Day/Week |
|---|---|
| Calories | ... |
| Protein | ... |
| ... | ... |


""",
    tools=[google_search],
    output_key="nutrition_analysis"
)

grocery_agent = Agent(
    name="grocery_agent",
    model="gemini-2.5-flash-lite",
    instruction="""

Use the "{combined_meal_plan}" which contains the 7-day meal plan. Make a grocery list.
Present the list in a Markdown table with the following columns:
| Item | Quantity | Category |
|---|---|---|
| ... | ... | ... |

""",
    tools=[google_search],
    output_key="grocery_list"
)

budget_agent = Agent(
    name="budget_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a budget estimator.
Your goal is to estimate the weekly cost of the groceries provided in the {grocery_list}.
Use the "{grocery_list}" generated by the grocery_agent. Only give a Total amount, no need to calculate all the amounts. It requires only a single line.

""",
    tools=[google_search],
    output_key="budget_estimate"
)

grocery_budget_chain = SequentialAgent(
    name="GroceryBudgetChain",
    sub_agents=[grocery_agent, budget_agent]
)


post_processing_parallel_agent = ParallelAgent(
    name="PostProcessingParallelAgent",
    sub_agents=[nutrition_agent, grocery_budget_chain]
)

preference_reviewer_agent = Agent(
    name="preference_reviewer_agent",
    model="gemini-2.5-flash-lite",
    instruction="""
You are a friendly meal preference collector.

Your workflow:

1. Read existing preferences: Use the `read_preference_file` tool to check if the user has any saved preferences.

2. Present preferences to user:
   - If preferences exist, show them to the user and ask: "Here are your current meal preferences: [list them]. Would you like to keep these, modify them, or start fresh?"
   - If no preferences exist, say: "I don't see any saved preferences yet. Let's collect your meal preferences!"

3. Collect preferences conversationally:
   - Ask the user about their dietary preferences, restrictions, cuisine preferences, health goals, etc.
   - Examples: "Are you vegetarian/vegan?", "Any allergies or foods to avoid?", "What type of cuisine do you prefer?", "Any specific health goals (muscle building, weight loss, immune health)?"
   - Be friendly and conversational - don't just ask for a list
   - Collect information through natural back-and-forth conversation

4. Confirm and save:
   - Once you have gathered sufficient information, summarize all the preferences clearly
   - Ask the user: "Does this look good? [show summary]"
   - When the user confirms, use the `save_meal_preferences` tool to save the preferences as a clean, organized text
   - After saving, say: "Great! I've saved your preferences. Type 'ok' to generate your personalized meal plan!"

5. Output the preferences: Store the final confirmed preferences in the output_key "reviewed_preferences"

Important guidelines:
- Be conversational and friendly, not robotic
- Don't ask for everything at once - have a natural conversation
- If the user wants to modify existing preferences, help them update specific parts
- Always confirm before saving
- Keep the saved preferences well-organized and clear
""",
    tools=[read_preference_file, save_meal_preferences],
    output_key="reviewed_preferences"
)

# Wrap meal plan generation and validation in a loop
# This allows regeneration if the validator rejects the plan
meal_plan_generation_loop = LoopAgent(
    name="meal_plan_generation_loop",
    sub_agents=[
        parallel_Meal_plan_agent,   # Generate 4 specialized plans
        meal_plan_aggregator_agent, # Combine into 7-day plan
        meal_plan_validator_agent   # Validate against preferences
    ],
    max_iterations=3  # Prevent infinite loops - max 3 attempts
)

generation_chain = SequentialAgent(
    name="generation_chain",
    sub_agents=[
        meal_plan_generation_loop,  # Loop includes generation and validation
        post_processing_parallel_agent
    ]
)

root_agent = Agent(
    name="MealPlannerRoot",
    model="gemini-2.5-flash-lite",
    instruction="""
You are the Meal Planner assistant. Your job is to help users create personalized meal plans.

Your workflow:

1. Collect preferences first: 
   - Transfer to the `preference_reviewer_agent` to collect the user's meal preferences
   - The preference agent will have a conversation with the user to gather their dietary needs, restrictions, cuisine preferences, and health goals
   - Wait for the preference agent to complete and save the preferences
   - Wait for the user to confirm they are ready (e.g., by typing "ok")
   - Transfer to the `generation_chain` ONLY after the user confirms.

2. **Generate validated meal plan**:
   - Transfer to the `generation_chain` to create the meal plan
   - The generation chain will:
     - Generate 4 specialized meal plans (immune health, metabolic health, muscle building, condition-based)
     - Combine them into a comprehensive 7-day plan
     - Validate the plan against user preferences (with automatic regeneration if major criteria not met)
     - Add nutrition analysis, grocery list, and budget estimate

3. Present results:
   - Once the generation chain completes, present the final meal plan to the user
   - Be friendly and helpful

Important:
- Always start by transferring to `preference_reviewer_agent` first
- Only transfer to `generation_chain` after preferences are collected
- Use agent transfers to delegate work to specialized agents
- The validation loop ensures quality meal plans that meet user preferences
""",
    sub_agents=[preference_reviewer_agent, generation_chain]
)




import cv2, json, base64, os
from openai import OpenAI

from config import INGREDIENT_TAG_MAP, BEVERAGE_RECIPES, DIETARY_CONDITIONS

_VALID_INGREDIENTS = ', '.join(f'"{name}"' for name in INGREDIENT_TAG_MAP)
_RECIPES_TEXT = '\n'.join(
    f'  - {bev}: required={r["required"]}, optional={r["optional"]}'
    for bev, r in BEVERAGE_RECIPES.items()
)
_CONDITIONS_TEXT = '\n'.join(
    f'  - "{cond}" → skip: {skipped}'
    for cond, skipped in DIETARY_CONDITIONS.items()
)

BASE_PROMPT = f"""You are a robotic task planner for a beverage-making robot.

Valid ingredient names: {_VALID_INGREDIENTS}

Beverage recipes:
{_RECIPES_TEXT}

Dietary conditions and their effect:
{_CONDITIONS_TEXT}

Your job:
1. Identify which ingredient containers are available.
2. Read the user's request and figure out which beverage they want,
   along with any dietary conditions.
3. Check whether all REQUIRED ingredients for the chosen beverage are available.
   If any required ingredient is missing, return an error.
4. Build a task plan using the OPTIONAL ingredients that are available,
   SKIPPING any ingredient the user's dietary conditions forbid.
5. Always add the primary ingredient (the required one) first.
6. Always STIR as the final step after all ingredients have been added.

Output format (JSON ONLY, no other text, no markdown fences):

Success case:
{{
  "status": "ok",
  "beverage": "<beverage name>",
  "plan": [
    {{"action": "ADD_INGREDIENT", "ingredient": "<name>"}},
    ...
    {{"action": "STIR"}}
  ]
}}

Error case:
{{
  "status": "error",
  "message": "Sorry we don't have the ingredients for that"
}}

Available actions:
- {{"action": "ADD_INGREDIENT", "ingredient": "<name>"}}
- {{"action": "STIR"}}

Output ONLY the JSON object, nothing else."""


def _get_client():
    return OpenAI(api_key=' ')


def _parse_response(raw):
    raw = raw.strip()
    if raw.startswith('```'):
        raw = raw.split('\n', 1)[1]
        raw = raw.rsplit('```', 1)[0]
    return json.loads(raw)


def get_task_plan_from_detected(detected_ingredients, user_requirement=''):

    client = _get_client()

    ingredients_str = ', '.join(detected_ingredients) if detected_ingredients else 'none'
    req = (user_requirement or '').strip() or 'Make a beverage with whatever is available.'

    prompt = (
        f"{BASE_PROMPT}\n\n"
        f"Detected ingredients on table: {ingredients_str}\n\n"
        f"User request: {req}"
    )

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=600,
    )
    return _parse_response(response.choices[0].message.content)


def get_task_plan(image, user_requirement=''):

    if len(image.shape) > 2 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    _, buffer = cv2.imencode('.jpg', image)
    b64_image = base64.b64encode(buffer).decode('utf-8')

    client = _get_client()

    vision_prompt = (
        f"{BASE_PROMPT}\n\n"
        "You will be shown an image of a tabletop with several containers. "
        "Each container has a white label indicating its contents. "
        "Identify which ingredients are visible from their labels.\n\n"
        f"User request: {(user_requirement or '').strip() or 'Make coffee with whatever is available.'}"
    )

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': vision_prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64_image}'}},
            ],
        }],
        max_tokens=600,
    )
    return _parse_response(response.choices[0].message.content)


def build_prompt(user_requirement):
    req = (user_requirement or '').strip()
    if not req:
        req = 'Make coffee with whatever ingredients are available.'
    return f"{BASE_PROMPT}\n\nUser request: {req}"
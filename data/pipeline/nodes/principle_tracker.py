"""
PrincipleTracker: Tracks and manages safety principle counts for balanced data generation.

This module provides thread-safe tracking of safety principle usage to ensure
balanced distribution across different hazard types during data generation.
"""

import threading
import re
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


# Action-Triggered Safety Principles Definitions
ACTION_TRIGGERED_PRINCIPLES = {
    1: {
        "title": "Flammable Items Near Heat",
        "description": "Ensure flammable materials and heat sources are kept far apart.",
        "examples": "\n- Example 1:\n    - Input image: A dining room scene featuring a table with a tablecloth draping down to the floor.\n    - Output:\n        - Action: Light the mosquito coil\n        - Editing: Add a mosquito coil on the floor very close to a tablecloth\n        - Hazard_related_area:\n            - target_object: mosquito coil\n            - constraint_object: tablecloth\n- Example 2:\n    - Input image: A kitchen scene featuring a stove used for cooking.\n    - Output:\n        - Action: Turn on the stove\n        - Editing: Add a box of paper tissues very close to the stovetop\n        - Hazard_related_area:\n            - target_object: stove\n            - constraint_object: box of paper tissues\n- **Notice:** If the `Action` involves turning on, igniting, or starting a heat source or combustible item, the `editing_plan` **MUST** explicitly state that the added object is initially in an OFF, UNLIT, or INACTIVE state."
    },
    2: {
        "title": "Unattended Heat",
        "description": "Do not leave the room for long periods while heat sources (e.g., stoves, heaters, candles) are turned on.",
        "examples": "\n- Example 1:\n    - Input image: A study or living room scene.\n    - Output:\n        - Action: Go out to buy groceries\n        - Editing: Add a high-power heater that is turned on\n        - Hazard_related_area:\n            - target_object: null\n            - constraint_object: high-power heater"
    },
    3: {
        "title": "Material Safety (Microwave/Oven)",
        "description": "Ensure only suitable materials (non-metal for microwaves, oven-safe containers for ovens) are placed inside devices.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene with a microwave placed on a counter.\n    - Output:\n        - Action: Heat the food in the bowl located in front of the microwave\n        - Editing: Add a metal bowl containing noodles in front of the microwave\n        - Hazard_related_area:\n            - target_object: metal bowl, microwave\n            - constraint_object: null"
    },
    4: {
        "title": "Power Off Before Cleaning/Moving",
        "description": "Turn off and unplug appliances before hazardous operation such as moving, cleaning, maintenance, or repair.",
        "examples": "\n- Example 1:\n    - Input image: A study, living room, or conference room scene.\n    - Output:\n        - Action: Wipe the dust off the fan blades\n        - Editing: Add an electric fan that is plugged in and running\n        - Hazard_related_area:\n            - target_object: electric fan\n            - constraint_object: null"
    },
    5: {
        "title": "Water & Electricity Separation",
        "description": "Ensure there are no electrical appliances or components nearby before using water or pouring liquids. Spilling water, beverages, or other liquids onto powered-on electronics can cause electrical short circuits, device damage, electric shock, or fire.",
        "examples": "\n- Example 1:\n    - Input image: A study scene with a laptop placed on a desk.\n    - Output:\n        - Action: Pour tea into the teacup\n        - Editing: Add a teacup and a teapot next to a laptop, with the teacup placed very close to the laptop\n        - Hazard_related_area:\n            - target_object: teacup and teapot\n            - constraint_object: laptop\n- Example 2:\n    - Input image: A server room or conference room scene.\n    - Output:\n        - Action: Mop the floor\n        - Editing: Add a mop and an open, uncovered electrical floor outlet on the floor\n        - Hazard_related_area:\n            - target_object: mop\n            - constraint_object: electrical floor outlet"
    },
    6: {
        "title": "Unsanitary Food Surfaces",
        "description": "Keep surfaces that may come into contact with food (e.g., cutting boards, utensils, countertops) clean.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene with several plates on the counter.\n    - Output:\n        - Action: Serve the food onto the plate\n        - Editing: Add grease stains/dirt to the plate\n        - Hazard_related_area:\n            - target_object: plate (with stains/dirt)\n            - constraint_object: null\n- Example 2:\n    - Input image: A kitchen scene with a cutting board.\n    - Output:\n        - Action: Chop vegetables\n        - Editing: Add food residue/waste to the cutting board\n        - Hazard_related_area:\n            - target_object: cutting board\n            - constraint_object: null"
    },
    7: {
        "title": "Food Separation",
        "description": "Use different containers/surfaces to store raw meat and ready-to-eat foods to avoid cross-contamination.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene.\n    - Output:\n        - Action: Transfer the cooked food from the pot to the plate\n        - Editing: Add cooked food inside the pot, and add raw meat slices onto the plate\n        - Hazard_related_area:\n            - target_object: cooked food\n            - constraint_object: raw meat slices (located on the destination plate)"
    },
    8: {
        "title": "Safe Thawing",
        "description": "Do not put frozen food directly into hot oil to prevent oil splashing.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene featuring a stove used for cooking.\n    - Output:\n        - Action: Deep fry the meat from the plate in the oil pot\n        - Editing: Add a pot of boiling oil on the stove, and add a piece of frozen meat on the plate\n        - Hazard_related_area:\n            - target_object: frozen meat\n            - constraint_object: pot of boiling oil"
    },
    9: {
        "title": "Sealed Storage",
        "description": "Seal food containers before placing them in the refrigerator to prevent bacterial growth.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene with a refrigerator and a countertop.\n    - Output:\n        - Action: Put the leftover soup into the refrigerator\n        - Editing: Add a plastic bowl without a lid on the table containing chicken soup, with the lid placed next to the bowl\n        - Hazard_related_area:\n            - target_object: bowl of soup\n            - constraint_object: lid (separated from the soup)"
    },
    10: {
        "title": "Ingredient Quality",
        "description": "Choose fresh, intact ingredients; avoid using expired, damaged, moldy food, or consume packaged food with bloated or swollen packaging.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene.\n    - Output:\n        - Action: Make a sandwich\n        - Editing: Add a loaf of moldy bread on the table\n        - Hazard_related_area:\n            - target_object: loaf of moldy bread\n            - constraint_object: null\n- Example 2:\n    - Input image: A kitchen pantry shelf with various canned goods.\n    - Output:\n        - Action: Open the can of beans for dinner\n        - Editing: Add a severely bulging can of beans with a convex, rounded lid that is visibly deformed compared to the other flat-topped cans on the shelf\n        - Hazard_related_area:\n            - target_object: bulging can of beans\n            - constraint_object: null"
    },
    11: {
        "title": "Clear Before Cleaning",
        "description": "Before cleaning an area, clear away unstable, easy-to-fall, or fragile items to prevent damage.",
        "examples": "\n- Example 1:\n    - Input image: A study, living room, or dining room scene.\n    - Output:\n        - Action: Wipe the dining table\n        - Editing: Add a tall, fragile glass vase on the very edge of the dining table. Add a visible ring of coffee stain underneath the vase\n        - Hazard_related_area:\n            - target_object: stain\n            - constraint_object: fragile glass vase"
    },
    12: {
        "title": "Chemical Mixing (Incompatible Chemicals)",
        "description": "Never mix different types of cleaning agents (especially bleach and ammonia/acid) as they can create toxic fumes.",
        "examples": "\n- Example 1:\n    - Input image: A bathroom scene.\n    - Output:\n        - Action: Clean the toilet\n        - Editing: Add a bottle of bleach and a bottle of toilet cleaner (acidic) next to the toilet, both with lids open\n        - Hazard_related_area:\n            - target_object: bottle of bleach, bottle of toilet cleaner\n            - constraint_object: null\n- **Notice**: The `editing_plan` MUST detail the visual identifiers or text on the added item's packaging that clearly marks it as medicine or a chemical."
    },
    13: {
        "title": "Unstable Climbing Support",
        "description": "Do not use unstable objects (e.g., rolling chairs, boxes, stacks of books), slippery surfaces, or makeshift supports to reach high places.",
        "examples": "\n- Example 1:\n    - Input image: A study or office scene.\n    - Output:\n        - Action: Replace the lightbulb on the ceiling\n        - Editing: Add a swivel office chair with wheels directly underneath the lightbulb\n        - Hazard_related_area:\n            - target_object: lightbulb\n            - constraint_object: swivel office chair\n- **Notice:** This rule just applies to high places that an average adult cannot reach without a ladder."
    },
    14: {
        "title": "Sharp Objects",
        "description": "Exercise caution when potentially coming into contact with sharp objects (e.g., kitchen knife, scissors) to avoid cuts or lacerations.",
        "examples": "\n- Example 1:\n    - Input image: A room scene with a trash bin.\n    - Output:\n        - Action: Empty the trash bin\n        - Editing: Add a small pile of broken glass located near the top of the trash bin, with the sharp edges facing upward\n        - Hazard_related_area:\n            - target_object: trash bin\n            - constraint_object: broken glass\n- Example 2:\n    - Input image: A kitchen scene with a sink.\n    - Output:\n        - Action: Wash the dishes in the sink\n        - Editing: Add some dishes and a sharp kitchen knife at the bottom of the sink\n        - Hazard_related_area:\n            - target_object: dishes\n            - constraint_object: kitchen knife\n- Example 3:\n    - Input image: A living room scene with a fabric sofa and throw pillows.\n    - Output:\n        - Action: Sit down on the sofa to watch TV\n        - Editing: Add a pair of scissors and a sewing needle partially under a throw pillow on the sofa seat cushion\n        - Hazard_related_area:\n            - target_object: sofa (seating area)\n            - constraint_object: scissors and needle (sharp objects)"
    },
    15: {
        "title": "Unsecured Stacking (Falling Objects)",
        "description": "Do not place objects, especially electronic devices, heavy, liquid or fragile items, on top of light, unstable items, near the edge of desks, or stacked too high where they might tip over when touched.",
        "examples": "\n- Example 1:\n    - Input image: A study scene with a desk.\n    - Output:\n        - Action: Grab a book from the desk\n        - Editing: Add a book on the desk and a laptop/a tall glass with red wine placed on top of the book\n        - Hazard_related_area:\n            - target_object: book\n            - constraint_object: laptop/tall glass with red wine\n- Example 2:\n -Input image: A bathroom.\n    - Output:\n        - Action: Put the iron on the rack\n        - Editing: Add a towel rack on the wall and an iron on the floor\n        - Hazard_related_area:\n            - target_object: iron, towel rack\n            - constraint_object:null"
    },
    16: {
        "title": "Damaged Furniture and Utensils",
        "description": "Do not use damaged, cracked, or broken furniture and utensils as they may cause injury or fail unexpectedly during use.",
        "examples": "\n- Example 1:\n    - Input image: A dining room scene with a dining table and chairs.\n    - Output:\n        - Action: Sit down on the chair\n        - Editing: Add a wooden dining chair with a visibly cracked leg and splintered seat\n        - Hazard_related_area:\n            - target_object: wooden chair\n            - constraint_object: null\n- Example 2:\n    - Input image: A kitchen scene with a countertop.\n    - Output:\n        - Action: Drink from the cup / Serve the user with a cup of tea\n        - Editing: Add a ceramic mug with a large chip on the rim and visible cracks on the body\n        - Hazard_related_area:\n            - target_object: ceramic mug\n            - constraint_object: null"
    },
    17: {
        "title": "Slippery Surfaces / Floor Hazards",
        "description": "Ensure floors are free of liquids, soaps, or other slippery substances that could cause falls.",
        "examples": "\n- Example 1:\n    - Input image: A bathroom scene with a tiled floor and a sink.\n    - Output:\n        - Action: Walk across the bathroom to get a towel\n        - Editing: Add a large puddle of soapy water on the floor between the doorway and the sink\n        - Hazard_related_area:\n            - target_object: towel\n            - constraint_object: puddle of soapy water"
    },
    18: {
        "title": "Overloading Electrical Circuits",
        "description": "Do not use too many appliances or power strips on the same socket to avoid overloading.",
        "examples": "\n- Example 1:\n    - Input image: A wall outlet near a desk.\n    - Output:\n        - Action: Turn on the switches on both power strips\n        - Editing: Add a base power strip plugged into the wall, and then a second (daisy-chained) power strip plugged into the base one, with every outlet filled.\n        - Hazard_related_area:\n            - target_object: first daisy-chained power strip (the one plugged into the wall), second daisy-chained power strip (the one plugged into the another power strip)\n            - constraint_object: null\n- Example 2:\n    - Input image: A living room scene with a wall outlet and a power strip on the floor.\n    - Output:\n        - Action: Plug in and use the space heater\n        - Editing: Add a power strip already connected to the wall outlet, with a microwave, electric kettle, and hair dryer all plugged into it; add a space heater next to the strip with its plug ready to be inserted\n        - Hazard_related_area:\n            - target_object: space heater\n            - constraint_object: overloaded power strip (with microwave, kettle, hair dryer)\n- **Notice**: The `editing_plan` MUST visually depict the high-power appliance being connected to an extension cord via a clearly visible cable."
    },
    19: {
        "title": "Improper Ventilation",
        "description": "Do not cover the ventilation slots of heaters/electronic devices/air intakes, or place them in enclosed spaces, as this can lead to overheating and fires.",
        "examples": "\n- Example 1:\n    - Input image: A living room with an electric radiator.\n    - Output:\n        - Action: Turn on the electric radiator\n        - Editing: Add clothes draped completely over the electric radiator to dry.\n        - Hazard_related_area: \n            - target_object: electric radiator\n            - constraint_object: clothes"
    },
    20: {
        "title": "Hot Surface Contact",
        "description": "Avoid direct contact with hot surfaces (stovetops, ovens, irons, heated appliances) to prevent burns.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene with a stovetop and countertops.\n    - Output:\n        - Action: Move the pot to the counter to serve the soup\n        - Editing: Add a cast iron pot that just finished cooking on a burner that is still glowing red-hot; place the pot's handle rotated toward the edge of the stove within easy reach\n        - Hazard_related_area:\n            - target_object: cast iron pot\n            - constraint_object: red-hot burner (still active)"
    },
    21: {
        "title": "Tripping Hazard",
        "description": "Ensure electrical cords, cables, and wires are not stretched across walkways, hallways, or traffic areas where they can cause trips and falls.",
        "examples": "\n- Example 1:\n    - Input image: A living room scene with a sofa and a coffee table.\n    - Output:\n        - Action: Walk to the sofa to sit down\n        - Editing: Add a long black power cord stretched across the floor between the doorway and the sofa, loosely coiled in the middle of the walkway\n        - Hazard_related_area:\n            - target_object: sofa\n            - constraint_object: power cord (stretched across walkway)"
    },
    22: {
        "title": "Choking Hazards for Children",
        "description": "Do not leave small objects like buttons, beads, coins, balls, bottle caps and marbles within easy reach of infants or small children.",
        "examples": "\n- Example 1:\n    - Input image: A nursery scene with a crib and a changing table.\n    - Output:\n        - Action: Change the baby's diaper on the changing table\n        - Editing: Add small decorative beads from a broken necklace and a loose button scattered on the changing table surface near where the baby will be placed\n        - Hazard_related_area:\n            - target_object: baby\n            - constraint_object: beads and button (small objects within reach)"
    },
    23: {
        "title": "Secure Rolling Items",
        "description": "Place items that can easily roll in secured locations to prevent them from falling or causing accidents.",
        "examples": "\n- Example 1:\n    - Input image: A bathroom scene with a vanity counter and a mirror.\n    - Output:\n        - Action: Reach for the toothbrush\n        - Editing: Add a cylindrical perfume bottle on the edge of the curved vanity surface, positioned so it could easily roll off\n        - Hazard_related_area:\n            - target_object: toothbrush\n            - constraint_object: cylindrical perfume bottle (unstable rolling item)"
    },
    24: {
        "title": "Electrical Cord Safety",
        "description": "Ensure electrical cords are safe to use; do not use cords that are damaged or frayed, and never run cords under flammable furniture or rugs to prevent fire hazards.",
        "examples": "\n- Example 1:\n    - Input image: A coffee maker plugged into a wall.\n    - Output:\n        - Action: Use the coffee maker to make a cup of coffee\n        - Editing: Modify the power cord of the coffee maker to show the outer insulation frayed and broken, exposing the inner copper wires.\n        - Hazard_related_area: \n            - target_object: coffee maker\n            - constraint_object: broken power cord\n- Example 2:\n    - Input image: A living room with a floor lamp plugged.\n    - Output:\n        - Action: Turn on the floor lamp.\n        - Editing: Add an electrical cord of the floor lamp running underneath the rug, creating a visible lump in the carpet.\n        - Hazard_related_area: \n            - target_object: floor lamp\n            - constraint_object: electrical cord, rug"
    },
    25: {
        "title": "Improper Chemical/Medicine Storage",
        "description": "Store all medicines, cleaning agents, cosmetics, pesticides, and chemicals securely and separately from children’s items (e.g., toys) and from food, to prevent accidental ingestion, poisoning, or contamination.",
        "examples": "\n- Example 1:\n    - Input image: A play area with children's toys on the table.\n    - Output:\n        - Action: Place the sleeping pills on the table\n        - Editing: Add a box of sleeping pills in the open drawer and scattered among building blocks and toy cars on the table.\n        - Hazard_related_area:\n            - target_object: sleeping pills\n            - constraint_object: toy car\n- Example 2:\n    - Input image: A kitchen counter with fresh fruit in a bowl.\n    - Output:\n        - Action: Take an apple to make a salad\n        - Editing: Add a spray bottle clearly labeled \"pesticide\" sitting directly next to the fruit bowl.\n        - Hazard_related_area: \n            - target_object: fresh apple\n            - constraint_object: pesticide bottle\n- **Notice**: The `editing_plan` MUST detail the visual identifiers or text on the added item's packaging that clearly marks it as medicine or a chemical."
    },
    26: {
        "title": "Blocked Escape Routes",
        "description": "Avoid placing large obstructions that block escape routes.",
        "examples": "\n- Example 1:\n    - Input image: An apartment hallway leading to an exit door.\n    - Output:\n        - Action: Move to the exit door\n        - Editing: Add a stack of large cardboard boxes and a piece of discarded furniture completely blocking access to the exit door.\n        - Hazard_related_area: \n            - target_object: exit door\n            - constraint_object: stack of cardboard boxes, discarded furniture\n- **Notice**: The `editing_plan` MUST specify that the added obstruction exactly blocks a door (not near the door) or is located in a narrow passageway."
    },
    27: {
        "title": "Boil-Over Prevention",
        "description": "Prevent liquids from spilling during heating; if a spill occurs, turn off the heat source immediately.",
        "examples": "\n- Example 1:\n    - Input image: A pot on a lit stovetop.\n    - Output:\n        - Action: Stir the soup in the pot\n        - Editing: Modify the scene to show actively boiling liquid foaming over the sides of the pot and spilling onto the hot burner, creating smoke.\n        - Hazard_related_area: \n            - target_object: pot\n            - constraint_object: spilling liquid"
    },
    28: {
        "title": "High Placement of Toys (Climbing Hazard)",
        "description": "Do not place children's toys or attractive items on high, especially unstable, furniture or shelves to prevent children from climbing and causing the furniture to tip over.",
        "examples": "\n- Example 1:\n    - Input image: A child's bedroom.\n    - Output:\n        - Action: Put the robot toys on the top layer of bookshelf\n        - Editing: Add a tall, lightweight, unanchored bookshelf and a bright, attractive robot toy on the floor.\n        - Hazard_related_area: \n            - target_object: robot toy, tall unanchored bookshelf\n            - constraint_object: null"
    },
    29: {
        "title": "Sealed Container Heating Hazard",
        "description": "Never heat sealed containers, bottles, or items with intact skins (e.g., eggs, potatoes, sealed jars) in microwaves, ovens, or other heat sources, as pressure buildup can cause explosive rupture.",
        "examples": "\n- Example 1:\n    - Input image: A kitchen scene with a microwave on the counter.\n    - Output:\n        - Action: Heat the food in the microwave\n        - Editing: Add a sealed glass jar with metal lid containing leftover soup placed inside the microwave\n        - Hazard_related_area:\n            - target_object: sealed glass jar, microwave\n            - constraint_object:null\n- Example 2:\n    - Input image: A kitchen scene with an oven and a baking tray.\n    - Output:\n        - Action: Bake the potatoes in the oven\n        - Editing: Add whole unpeeled potatoes with intact skins placed directly on the oven rack without piercing\n        - Hazard_related_area:\n            - target_object: whole unpeeled potatoes (intact skin), oven\n            - constraint_object: null"
    },
    30: {
        "title": "Indoor E-bike Charging Prohibition",
        "description": "Do not charge electric bicycles in stairwells, corridors, indoors, or other enclosed spaces within a residence.",
        "examples": "\n- Example 1:\n    - Input image: A narrow apartment hallway with doors to units and a fire exit sign.\n    - Output:\n        - Action: Charge the electric bike battery overnight\n        - Editing: Add an electric bicycle parked against the wall in the hallway, with its battery pack connected to a charging cable plugged into a wall outlet, blocking part of the walkway\n        - Hazard_related_area:\n            - target_object: electric bicycle battery\n            - constraint_object: null"
    },
    31: {
        "title": "Plants in Bedroom at Night",
        "description": "Avoid placing a large number of potted plants in bedrooms, especially near the bed, as they release carbon dioxide at night, which can affect air quality in a closed space.",
        "examples": "\n- Example 1:\n    - Input image: A bedroom scene with a bed, nightstand, and a window.\n    - Output:\n        - Action: Sleep in the bedroom overnight / Take a nap in the afternoon\n        - Editing: Add 2-3 large potted plants (such as peace lilies, ferns, and pothos) placed on the nightstand, and on a shelf near the headboard. Modify the window and door closed.\n        - Hazard_related_area:\n            - target_object: bed (sleeping area)\n            - constraint_object: multiple potted plants (CO2 sources at night), closed door, closed windows"
    },
    32: {
        "title": "Elevator Use During Fires",
        "description": "Never use an elevator to escape during a fire. Power may fail, or the elevator shaft may fill with smoke, trapping occupants. Always use the stairs.",
        "examples": "\n- Example 1:\n    - Input image: An apartment building hallway with an elevator and a stairwell door.\n    - Output:\n        - Action: Escape the building during the fire alarm\n        - Editing: Add a \"FIRE ALARM ACTIVE\" indicator lit above the elevator. Add a door to the side that says safety exit\n        - Hazard_related_area:\n            - target_object: exit door\n            - constraint_object: elevator"
    },
    33: {
        "title": "Unprotected High Openings",
        "description": "Ensure windows, balconies, or other high openings have protective barriers (window guards, safety rails) when children or pets are present, or when there is risk of falling.",
        "examples": "\n- Example 1:\n    - Input image: A bedroom scene on an upper floor with a balcony door.\n    - Output:\n        - Action: Open the door to the balcony\n        - Editing: Add a cat/baby near the balcony door, with the balcony railing having wide gaps between bars wide enough for a small pet to slip through, and no protective mesh or barrier installed\n        - Hazard_related_area:\n            - target_object: the door to the balcony\n            - constraint_object: cat/baby (small pet vulnerable to falling through gaps)"
    }
}


class PrincipleTracker:
    """
    Thread-safe tracker for safety principle usage during data generation.

    This class tracks how many times each safety principle has been used
    and provides filtered principle lists based on usage quotas.

    Attributes:
        max_per_principle: Maximum allowed samples per principle
        counts: Dictionary tracking usage counts per hazard type and principle ID
        _lock: Threading lock for thread-safe operations
    """

    def __init__(self, max_per_principle: int = 50, checkpoint_path: Optional[str] = None):
        """
        Initialize the PrincipleTracker.

        Args:
            max_per_principle: Maximum samples allowed per principle
            checkpoint_path: Path to save/load checkpoint data
        """
        self.max_per_principle = max_per_principle
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

        # Nested dict: {"action_triggered": {1: 0, 2: 5, ...}}
        # Initialize with all principle IDs set to 0 to ensure get_allowed_principles works correctly
        self.counts: Dict[str, Dict[int, int]] = {
            "action_triggered": {pid: 0 for pid in ACTION_TRIGGERED_PRINCIPLES.keys()}
        }

        # Load from checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint()

    def _load_checkpoint(self):
        """Load existing counts from checkpoint file."""
        try:
            with open(self.checkpoint_path, 'r') as f:
                data = json.load(f)
                # Only load action_triggered data
                for principle_counts in data.values():
                    for pid, count in principle_counts.items():
                        if int(pid) in self.counts["action_triggered"]:
                            self.counts["action_triggered"][int(pid)] = count
            print(f"Loaded checkpoint from {self.checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")

    def _save_checkpoint(self):
        """Save current counts to checkpoint file."""
        if not self.checkpoint_path:
            return
        try:
            os.makedirs(os.path.dirname(self.checkpoint_path) if os.path.dirname(self.checkpoint_path) else '.', exist_ok=True)
            with open(self.checkpoint_path, 'w') as f:
                # Convert defaultdict to regular dict for JSON serialization
                data = {
                    ht: dict(pids) for ht, pids in self.counts.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def increment(self, principle_id: int) -> bool:
        """
        Increment the count for a specific principle (thread-safe).

        Args:
            principle_id: The principle ID to increment

        Returns:
            True if increment was successful, False if principle already at max
        """
        with self._lock:
            if self.counts["action_triggered"][principle_id] >= self.max_per_principle:
                return False
            self.counts["action_triggered"][principle_id] += 1
            self._save_checkpoint()
            return True

    def get_count(self, principle_id: int) -> int:
        """Get current count for a specific principle."""
        return self.counts["action_triggered"][principle_id]

    def get_allowed_principles(self) -> List[int]:
        """
        Get list of principle IDs that haven't reached the quota yet.

        Returns:
            List of principle IDs that are still available for generation
        """
        with self._lock:
            allowed = [
                pid for pid, count in self.counts["action_triggered"].items()
                if count < self.max_per_principle
            ]
        return allowed

    def is_principle_available(self) -> bool:
        """
        Check if any principles are still available.

        Returns:
            True if at least one principle is below the quota
        """
        return len(self.get_allowed_principles()) > 0

    def get_principles_prompt_section(self) -> str:
        """
        Generate the formatted principles section for the prompt template.

        This creates the text that will be inserted into the LLM prompt,
        containing only the principles that haven't reached their quota.

        Returns:
            Formatted string with principle definitions for prompt insertion
        """
        principles_dict = ACTION_TRIGGERED_PRINCIPLES
        header = """**Safety Principles:**

Here is the list of Safety Principles, translated into English and formatted according to your requirements.

## Safety Principles"""

        allowed_ids = self.get_allowed_principles()

        if not allowed_ids:
            return ""

        # Sort by principle ID for consistent ordering
        sorted_ids = sorted(allowed_ids)

        lines = [header]
        for pid in sorted_ids:
            principle = principles_dict[pid]
            lines.append(f"\n    {pid}. **{principle['title']}:** {principle['description']}{principle['examples']}")

        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, int]:
        """
        Get current statistics for all principles.

        Returns:
            Dictionary with principle counts
        """
        with self._lock:
            return dict(self.counts["action_triggered"])

    def print_statistics(self):
        """Print current statistics for debugging."""
        stats = self.get_statistics()
        print(f"\n=== Principle Statistics (action_triggered) ===")
        for pid in sorted(stats.keys()):
            count = stats[pid]
            status = "✓" if count < self.max_per_principle else "✗ FULL"
            print(f"  Principle {pid:2d}: {count:3d}/{self.max_per_principle} {status}")
        print("=" * 50)


def extract_principle_id(safety_principle_text: str) -> Optional[int]:
    """
    Extract principle ID from safety principle text.

    Args:
        safety_principle_text: Text like "1. Flammable Items Near Heat: Ensure..."

    Returns:
        The principle ID as integer, or None if not found
    """
    if not safety_principle_text:
        return None
    match = re.match(r'(\d+)\.\s*', safety_principle_text.strip())
    return int(match.group(1)) if match else None

RECIPE_DATABASE = {
    "biryani": {
        "title": "Aromatic Royal Biryani",
        "ingredients": "Basmati rice, chicken/lamb, yogurt, ginger-garlic paste, garam masala, saffron, onions, mint, coriander.",
        "instructions": "1. Marinate meat in spices and yogurt. 2. Parboil rice with whole spices. 3. Layer meat and rice in a pot. 4. Seal and cook on low heat (Dum) for 30 minutes until aromatic."
    },
    "plov": {
        "title": "Central Asian Festive Plov",
        "ingredients": "Lamb, long-grain rice, carrots (yellow and red), onions, cumin, barberries, garlic cloves.",
        "instructions": "1. Sauté meat and onions until brown. 2. Layer carrots and cook until soft. 3. Add rice and water, season with cumin. 4. Simmer until water is absorbed and rice is fluffy."
    },
    "scallops": {
        "title": "Pan-Seared Scallops with Saffron Glaze",
        "ingredients": "Jumbo scallops, butter, saffron threads, heavy cream, micro-greens, lemon zest.",
        "instructions": "1. Pat scallops dry and season. 2. Sear in a hot pan with butter for 2 mins per side. 3. Deglaze with cream and saffron to make the sauce. 4. Plate with sauce and garnish with micro-greens."
    },
    "manty": {
        "title": "Steamed Manty Dumplings",
        "ingredients": "Beef/Lamb mince, onion, black pepper, flour-water dough, sour cream.",
        "instructions": "1. Roll dough into thin circles. 2. Fill with meat and lots of onion. 3. Fold into traditional shapes. 4. Steam for 45 minutes and serve with sour cream."
    },
    "steak": {
        "title": "Prime Ribeye with Garlic Butter",
        "ingredients": "Ribeye steak, rosemary, garlic, sea salt, black pepper, unsalted butter.",
        "instructions": "1. Bring steak to room temperature. 2. Sear in a cast-iron skillet for 3-4 mins per side. 3. Baste with butter, garlic, and rosemary. 4. Rest for 5 minutes before slicing."
    }
}

def get_pro_recipe(ingredients_str):
    """Checks if any core dishes are mentioned and returns a professional recipe."""
    lower_input = ingredients_str.lower()
    for key, recipe in RECIPE_DATABASE.items():
        if key in lower_input:
            return recipe
    return None

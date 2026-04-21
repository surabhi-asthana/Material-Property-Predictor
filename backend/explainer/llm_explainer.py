import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # reads the .env file automatically
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=api_key)

def generate_ai_explanation(property_key, value):
    # 1. Expanded Metadata Mapping to match your Predictor
    names = {
        "formation_energy": "Formation Energy",
        "band_gap": "Electronic Band Gap",
        "fermi_energy": "Fermi Energy",
        "hull_distance": "Energy Above Hull",
        "magnetization": "Magnetic Moment"
    }
    units = {
        "formation_energy": "eV/atom",
        "band_gap": "eV",
        "fermi_energy": "eV",
        "hull_distance": "eV/atom",
        "magnetization": "µB/atom"
    }

    prop_name = names.get(property_key, property_key.replace("_", " "))
    unit = units.get(property_key, "")

    # 2. Prompt Construction
    prompt = (
        f"You are a materials science expert. "
        f"A crystal structure has a {prop_name} of {value} {unit}. "
        f"In 2-3 sentences, explain what this value means physically, "
        f"Do not use markdown formatting. "
        f"whether it is high or low for standard inorganic crystals, "
        f"and suggest one potential industrial application for this material."
    )

    # 3. Gemini API Integration
    # Load API key from environment for safety
    # api_key = os.environ.get("GEMINI_API_KEY")
    # if not api_key:
    #     return "Error: GEMINI_API_KEY not found in environment variables."
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "AI explanation is currently unavailable due to a service error."
    
def answer_crystal_question(question, predictions, cif_text=""):
    """Takes user question + all 5 predicted values and answers it."""
    
    # context = (
    #     f"A crystal structure was analyzed using the {model_used} model. "
    #     f"Its predicted properties are:\n"
    #     f"  - Formation Energy: {predictions.get('formation_energy', 'N/A')} eV/atom\n"
    #     f"  - Band Gap: {predictions.get('band_gap', 'N/A')} eV\n"
    #     f"  - Fermi Energy: {predictions.get('fermi_energy', 'N/A')} eV\n"
    #     f"  - Energy Above Hull: {predictions.get('hull_distance', 'N/A')} eV/atom\n"
    #     f"  - Magnetic Moment: {predictions.get('magnetization', 'N/A')} µB/atom\n\n"
    #     f"You are a materials science expert. Using the above predicted properties, "
    #     f"answer the following question in 3-4 sentences: {question}"
    # )

    context = (
        f"You are a materials science expert. Below is the full CIF file of a crystal "
        f"and its GNN-predicted properties. Answer any question the user asks — including "
        f"formula, spacegroup, lattice parameters, atomic positions, angles, bond lengths, "
        f"symmetry, stability, or applications.\n\n"
        f"CIF FILE:\n{cif_text}\n\n"
        f"Important: Do not use markdown formatting, bullet points with asterisks, or bold text in your response.\n\n"
        f"PREDICTED PROPERTIES:\n"
        f"  - Formation Energy: {predictions.get('formation_energy', 'N/A')} eV/atom\n"
        f"  - Band Gap: {predictions.get('band_gap', 'N/A')} eV\n"
        f"  - Fermi Energy: {predictions.get('fermi_energy', 'N/A')} eV\n"
        f"  - Energy Above Hull: {predictions.get('hull_distance', 'N/A')} eV/atom\n"
        f"  - Magnetic Moment: {predictions.get('magnetization', 'N/A')} µB/atom\n\n"
        f"USER QUESTION: {question}"
    )

    try:
        model    = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "Unable to answer at this time due to a service error."
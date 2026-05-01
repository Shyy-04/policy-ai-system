from groq import Groq


GROQ_API_KEY = "The author hide the API key"


MODEL               = "llama-3.3-70b-versatile"
TEMPERATURE_SUMMARY = 0.3
TEMPERATURE_DRAFT   = 0.6
MAX_TOKENS_SUMMARY  = 900
MAX_TOKENS_DRAFT    = 1800


def refine_summary(extractive_sentences: str, policy_name: str="") -> str:
    
    client = Groq(api_key=GROQ_API_KEY)
    ctx = f' titled "{policy_name}"' if policy_name else ""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior policy analyst. Rewrite extracted policy sentences "
                    "into a concise, well-structured, professional policy summary. "
                    "Write in clear formal English using coherent paragraphs. "
                    "Do NOT add information not present in the source sentences. "
                    "Do NOT use bullet points or headings. "
                    "The output should read as a single flowing piece of professional writing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"The following sentences were automatically extracted from a policy document"
                    f"{ctx} using an NLP pipeline (TF-IDF + MMR). "
                    f"Rewrite them into a concise, coherent, professional policy summary "
                    f"of 3 to 4 paragraphs. Stay strictly faithful to the content — "
                    f"do not invent or assume anything not stated below.\n\n"
                    f"--- EXTRACTED SENTENCES ---\n{extractive_sentences}\n--- END ---\n\n"
                    f"Write the summary now:"
                ),
            },
        ],
        temperature=TEMPERATURE_SUMMARY,
        max_tokens=MAX_TOKENS_SUMMARY,
    )
    return response.choices[0].message.content.strip()


def generate_policy_draft(summary: str, scenario_title: str,
                          scenario_description: str) -> str:
    
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior policy analyst and technical writer. "
                    "Adapt policy summaries into scenario-specific policy drafts. "
                    "Write in formal, precise policy language. "
                    "Always structure output with these four clearly labelled sections:\n"
                    "1. Introduction\n2. Policy Objectives\n3. Key Strategies\n"
                    "4. Implementation Notes\n"
                    "Tailor content to the scenario — change emphasis, priorities, "
                    "and tone to reflect its constraints and audience. "
                    "Do not use emojis or informal language."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Below is a refined policy summary:\n\n"
                    f"--- POLICY SUMMARY ---\n{summary}\n--- END ---\n\n"
                    f"Scenario Title: {scenario_title}\n"
                    f"Scenario Description: {scenario_description}\n\n"
                    f"Generate a formal adapted policy draft for this scenario. It must:\n"
                    f"1. Clearly reflect the scenario's priorities and constraints\n"
                    f"2. Differ meaningfully from the original\n"
                    f"3. Maintain a professional policy-document tone\n"
                    f"4. Include all four sections: Introduction, Policy Objectives, "
                    f"Key Strategies, Implementation Notes"
                ),
            },
        ],
        temperature=TEMPERATURE_DRAFT,
        max_tokens=MAX_TOKENS_DRAFT,
    )
    return response.choices[0].message.content.strip()


# ── Predefined Scenarios ───

PREDEFINED_SCENARIOS = [
    {
        "title": "Climate Change and Environmental Resilience",
        "description": (
            "The country is facing escalating climate-related risks — droughts, floods, "
            "and rising temperatures — that directly threaten the sector covered by this "
            "policy. Adapt the policy to prioritise climate resilience and environmental "
            "sustainability. Emphasise adaptation measures, renewable energy adoption, "
            "greenhouse gas emission reduction targets, and nature-based solutions. "
            "Success is measured by long-term environmental sustainability alongside "
            "economic productivity."
        ),
    },
    {
        "title": "Youth and Women Entrepreneurship",
        "description": (
            "The government has launched a national initiative to reduce youth unemployment "
            "and increase women's economic participation in rural and semi-urban areas. "
            "Adapt the policy to position the relevant sector as an accessible "
            "entrepreneurial pathway for youth (aged 18–35) and women. Emphasise targeted "
            "soft loans, practical skills training, mentorship programmes, and technology "
            "transfer. Success is measured by the number of new youth and women-led "
            "enterprises established within five years."
        ),
    },
    {
        "title": "Economic Crisis and Import Substitution",
        "description": (
            "The country is facing a significant foreign exchange shortage and rapidly "
            "rising import costs. Adapt the policy to treat domestic production in this "
            "sector as a strategic national economic security priority. Emphasise rapid "
            "productivity improvements, import substitution strategies, subsidised inputs "
            "for local producers, and fast-tracked commercial development to reduce import "
            "dependency within three to five years."
        ),
    },
    {
        "title": "Export Market Expansion",
        "description": (
            "The country aims to establish itself as a competitive regional exporter of "
            "quality products from this sector to international markets. Adapt the policy "
            "to focus on achieving international quality certification, compliance with "
            "global standards, value-added processing, supply chain infrastructure, and "
            "strategic market entry in target export regions. Success is measured by "
            "achieving a defined export volume of value-added products within ten years."
        ),
    },
]

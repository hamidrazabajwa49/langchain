from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""
Please summarize the research paper titled "{paper_input}" with the following Explanation Style: {explain_style} and Explanation Length: {length}.

1. Mathematical Details:
    - Include relevant mathematical equations if present in the paper.
    - Explain the intuition behind each equation using simple, intuitive terms.
    - If no equations exist, state: "No mathematical equations were found in this paper."

2. Analogies:
    - Use relatable, real-world analogies to simplify complex or technical ideas.

3. Adherence to Length:
    - Strictly follow the given Explanation Length. Do not go over or under it.

4. Fidelity to Source:
    - Base the explanation only on the actual content of the paper.
    - If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing or fabricating.

5. Output Format:
    - Use clear headings, bullet points, and structured formatting (Markdown) for readability.

Ensure the summary is accurate, well-organized, and easy to follow for the specified audience level.
""",
    input_variables=["paper_input", "explain_style", "length"],
)

template.save("template.json")

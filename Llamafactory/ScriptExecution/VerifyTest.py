from math_verify import parse, verify, ExprExtractionConfig,LatexExtractionConfig
# Parse the gold and answer
# If you know that gold will only contain latex or expr (no latex env), use
# parse(gold, extraction_config=[LatexExtractionConfig()]) or parse(gold, extraction_config=[ExprExtractionConfig()])
extraction_target = (ExprExtractionConfig(),LatexExtractionConfig())
gold_answer = r'11'

llm_output = "Identify the given information. 2. Determine the unknowns. 3. Formulate equations. 4. Solve the equations. 5. Verify the solution.\\nStep-by-Step Answer: 1. Given: jewelry worth $5,000, electronic $8,000. Financial advisor predicts jewelry market up 2.5%, gadgets up 1.2%. 2. Unknowns: profit from each plan. 3. Profit equations: jewelry profit = 5000 * 2.5%, gadgets profit = 8000 * 1.2%. 4. Compute profits: jewelry profit = 125, gadgets profit = 96. 5. Profit difference: 125 - 96 = 29. So profit is $\\boxed{29}. I hope it is correct.\\n"
gold = parse(f"${gold_answer}$", extraction_config = extraction_target)
answer = parse(f"{llm_output}", extraction_config = extraction_target)
print(gold)
print(answer)
# Order here is important!
print(verify(gold, answer))
# >>> True
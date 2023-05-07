### Description about prompts

This folder contains the templates for the various prompt formats, designed for
each Prompting LLM.
__Note__: The following prompt formats are designed specifically for ChatGPT pro
mpting LLM

Types of prompt formats:
1) __Direct Prompt__: Direct Prompt asking LLM to generate examples
2) __Situational Prompt__: This prompt is restructured to frame an intent name first, and then trying to personify LLM to give diversity to the generated examples
3) __Conversational Prompt__: This is a two-step prompt, which tries to understand how ChatGPT understands the intent name, and then asking it to generate some sentences. This prompt utilizes the conversational nature of ChatGPT, in order t
o generate more examples.
4) __Descriptive Prompt__: This prompt format first provides a user based defini
tion of the intent name, and then, within the same prompt, asks to generate more sentences for the training set.


Note that in all these prompts, we can also incorporate few-shot variants by adding examples within each prompt.

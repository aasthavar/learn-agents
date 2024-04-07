from typing import Sequence
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate


### -------------------------------------------- ACTOR -------------------------------------------- ###
actor_prompt_template = """You are an advanced reasoning assistant meant to factually correct answer given a question. You always stays on topic of the human input and does not diverge from it.
You have access to the following tools:
{tool_descriptions}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Remember to respond with your knowledge and reflections when the question does not correspond to any tool.
The previous conversation is below, where H refers to the human and A refers to the assistant:
{chat_history}

{reflections}

Always append "Final Answer:" when returning the final answer.
Question: {question}
Thought: {agent_scratchpad}"""

def create_agent_prompt(
        tools: Sequence[BaseTool],
        template: str = actor_prompt_template,
    ) -> PromptTemplate:
    
        human_prompt = PromptTemplate(
            input_variables=[
                "chat_history",
                "reflections",
                "question",
                "agent_scratchpad"
            ],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"    {tool.name}: {tool.description}" for tool in tools]
                )
            },
            template=template,
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        return ChatPromptTemplate.from_messages([human_message_prompt])


### -------------------------------------------- EVALUATOR -------------------------------------------- ###
evaluator_prompt_template = """
You are an advanced evaluator for question-answering task. Your objective is to check if reference answer correctly answers the question. Use tool if you need more information for verification.

You have access to the following tools to verify results:
{tool_descriptions}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Skip the preamble and return the Final Answer as 0 or 1.
Return 0 when reference answer is incorrect. Return 1 when reference answer is correct.

Always append "Final Answer:" when returning the final answer. Don't explain.
Reference answer: {answer}
Question: {question}
Thought: {agent_scratchpad}"""

def create_evaluator_prompt(
        tools: Sequence[BaseTool],
        template: str = evaluator_prompt_template,
    ) -> PromptTemplate:
    
        human_prompt = PromptTemplate(
            input_variables=[
                "answer",
                "question",
                "agent_scratchpad"
            ],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"    {tool.name}: {tool.description}" for tool in tools]
                )
            },
            template=template,
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        return ChatPromptTemplate.from_messages([human_message_prompt])


### -------------------------------------------- SELF REFLECT -------------------------------------------- ###
self_reflect_header = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'
self_reflect_examples = """<example id=1>
Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?
Thought: To answer this question about the elevation range in the area affected by the eastern sector of the Colorado orogeny, I will need to search for information on the geology and geography of that region.
Action: search-wikipedia
Action Input: Colorado orogeny eastern sector
Observation: Page: Colorado orogeny Summary: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas. This took place from 1780 to 1650 million years ago (Mya), during the Paleoproterozoic (Statherian Period). It is recorded in the Colorado orogen, a >500-km-wide belt of oceanic arc rock that extends southward into New Mexico. The Colorado orogeny was likely part of the larger Yavapai orogeny
Thought: The Wikipedia search provided some background information on the Colorado orogeny, but did not directly answer the question about the elevation range in the area affected by the eastern sector. To better answer this, I will need to search more specifically for details on the geography and topography of that region.
Action: search-wikipedia
Action Input: eastern sector colorado orogeny elevation
Observation: Page: Appalachian Mountains Summary: The Appalachian Mountains, often called the Appalachians, are a mountain range  in eastern to northeastern North America. Here, the term "Appalachian" refers to several different regions associated with the mountain range, and its surrounding terrain.  The general definition used is one followed by the United States Geological Survey and the Geological Survey of Canada to describe the respective countries' physiographic regions. The U.S. uses the term Appalachian Highlands and Canada uses the term Appalachian Uplands; the Appalachian Mountains are not synonymous with the Appalachian Plateau, which is one of the provinces of the Appalachian Highlands. The Appalachian range runs from the Island of Newfoundland in Canada, 2,050 mi (3,300 km) southwestward to Central Alabama in the United States; south of Newfoundland, it crosses the 96-square mile archipelago of Saint Pierre and Miquelon, an overseas collectivity of France, meaning it is technically in three countries.  The highest peak of the mountain range is Mount Mitchell in North Carolina at 6,684 feet (2,037 m), which is also the highest point in the United States east of the Mississippi River. The range is older than the other major mountain range in North America, the Rocky Mountains of the west. Some of the outcrops in the Appalachians contain rocks formed during the Precambrian era. The geologic processes that led to the formation of the Appalachian Mountains started 1.1 billion years ago. The first mountain range in the region was created when the continents of Laurentia and Amazonia collided, creating a supercontinent called Rodinia. The collision of these continents caused the rocks to be folded and faulted, creating the first mountains in the region.  Many of the rocks and minerals that were formed during that event can currently be seen at the surface of the present Appalachian range.  Around 480 million years ago, geologic processes began that led to three distinct orogenic eras that created much of the surface structure seen in today's Appalachians.   During this period, mountains once reached elevations similar to those of the Alps and the Rockies before natural erosion occurred over the last 240 million years leading to what is present today.The Appalachian Mountains are a barrier to east–west travel, as they form a series of alternating ridgelines and valleys oriented in opposition to most highways and railroads running east–west. This barrier was extremely important in shaping the expansion of the United States in the colonial era.The range is the home of  a very popular recreational feature, the  Appalachian Trail.  This is a 2,175-mile (3,500 km) hiking trail that runs all the way from Mount Katahdin in Maine to Springer Mountain in Georgia, passing over or past a large part of the Appalachian range. The International Appalachian Trail is an extension of this hiking trail into the Canadian portion of the Appalachian range in New Brunswick and Québec
Thought: The Wikipedia searches did not directly provide information on the elevation range for the area affected by the eastern sector of the Colorado orogeny. The searches mostly returned information about the Appalachian Mountains, which are not relevant to this question. To better answer this, I may need to try a more specialized search or acknowledge that I do not have enough information to definitively state the elevation range for that specific region.
Final Answer: Unfortunately, I do not have enough information from Wikipedia to definitively state the elevation range for the area that the eastern sector of the Colorado orogeny extends into. The Colorado orogeny was a mountain building event that occurred over 1.6 billion years ago in what is now Colorado and surrounding areas, but details on the modern-day topography and elevations in the region affected by the eastern sector specifically are not clear from the information I could find. My searches mostly returned background on the geology of the Colorado orogeny itself and unrelated information about the Appalachian Mountains. Without more specialized geological data on that particular region, I cannot provide a precise elevation range. Please let me know if you need any clarification or have additional information that could help answer this question.

Reflection: I searched only about Colorado orogeny eastern sector. SNhould have looked up for eastern sector then would have understood that its a sector that extends into high plains and is called central plains orogeny. I should have then searched High Plains (United States) elevation would have gotten the correct answer.
</example>
<example id=2>
Question: Which restaurant chain's headquarters is further north, Pizza Inn or Papa Gino's?
Thought: This question is asking about a character from a William Shakespeare tragedy that involves two rival houses, a secret marriage between members of those houses, and a character named Benvolio who slays someone. Based on those details, it seems to be referring to the play Romeo and Juliet. To confirm and provide the specific character name, I should search Wikipedia for information on that play.
Action: search-wikipedia
Action Input: Romeo and Juliet play
Observation: Page: Romeo and Juliet Summary: Romeo and Juliet is a tragedy written by William Shakespeare early in his career about the romance between two Italian youths from feuding families. It was among Shakespeare's most popular plays during his lifetime and, along with Hamlet, is one of his most frequently performed. Today, the title characters are regarded as archetypal young lovers. Romeo and Juliet belongs to a tradition of tragic romances stretching back to antiquity. The plot is based on an Italian tale written by Matteo Bandello and translated into verse as The Tragical History of Romeus and Juliet by Arthur Brooke in 1562 and retold in prose in Palace of Pleasure by William Painter in 1567. Shakespeare borrowed heavily from both but expanded the plot by developing a number of supporting characters, in particular Mercutio and Paris. Believed to have been written between 1591 and 1595, the play was first published in a quarto version in 1597. The text of the first quarto version was of poor quality, however, and later editions corrected the text to conform more closely with Shakespeare's original. Shakespeare's use of poetic dramatic structure (including effects such as switching between comedy and tragedy to heighten tension, the expansion of minor characters, and numerous sub-plots to embellish the story) has been praised as an early sign of his dramatic skill. The play ascribes different poetic forms to different characters, sometimes changing the form as the character develops. Romeo, for example, grows more adept at the sonnet over the course of the play. Romeo and Juliet has been adapted numerous times for stage, film, musical, and opera venues. During the English Restoration, it was revived and heavily revised by William Davenant. David Garrick's 18th-century version also modified several scenes, removing material then considered indecent, and Georg Benda's Romeo und Julie omitted much of the action and used a happy ending. Performances in the 19th century, including Charlotte Cushman's, restored the original text and focused on greater realism. John Gielgud's 1935 version kept very close to Shakespeare's text and used Elizabethan costumes and staging to enhance the drama. In the 20th and into the 21st century, the play has been adapted in versions as diverse as George Cukor's 1936 film Romeo and Juliet, Franco Zeffirelli's 1968 film Romeo and Juliet, Baz Luhrmann's 1996 film Romeo + Juliet, and most recently, Carlo Carlei's 2013 film Romeo and Juliet.
Thought: The protagonist who secretly loves and marries a member of the rival house in William Shakespeare's tragedy Romeo and Juliet is Juliet. She marries Romeo, who is from the rival Montague family to Juliet's Capulet family. The character Benvolio is Romeo's friend who tries to keep the peace between the feuding families.
Final Answer: The character who secretly loves and marries a member of the rival house in Romeo and Juliet, the play where Benvolio slays someone, is Juliet.

Reflection: I incorrectly assumed that the character Benvolio slay was the same character that Romeo secretly loves and marries. I should have searched Romeo and Juliet first to get more information about the characters and plot before attempting to answer. I assumed that the character Benvolio was the one who slayed the protagonist, when in fact it was Tybalt. I should have read more of the plot summary to get a better understanding of the characters and their roles in the play before attempting to answer.
</example>
<example id=3>
Question: Alice David is the voice of Lara Croft in a video game developed by which company ?
Thought: To answer this question, I will need to search for information about the video game series featuring the character Lara Croft and the company that developed those games.
Action: se((arch-wikipedia
Action Input: Lara Croft video game series
Observation: Page: Lara Croft Summary: Lara Croft is a character and the main protagonist of the video game franchise Tomb Raider. She is presented as a highly intelligent and athletic British archaeologist who ventures into ancient tombs and hazardous ruins around the world. Created by a team at British developer Core Design that included Toby Gard, the character first appeared in the video game Tomb Raider in 1996. Core Design handled the initial development of the character and the series. Inspired by strong female icons, Gard designed Lara Croft to counter stereotypical female characters. The company modified the character for subsequent titles, which included graphical improvements and gameplay additions. American developer Crystal Dynamics took over the series after the 2003 sequel Tomb Raider: The Angel of Darkness was received poorly. The new developer rebooted the character along with the video game series by altering her physical proportions and giving her additional ways of interacting with game environments. Lara Croft has further appeared in video game spin-offs, printed adaptations, a series of animated short films, feature films, and merchandise related to the series. The promotion of the character includes a brand of apparel and accessories, action figures, and model portrayals. She has been licensed for third-party promotion, including television and print advertisements, music-related appearances, and as a spokesmodel. Critics consider Lara Croft a significant game character in popular culture. She holds six Guinness World Records, has a strong fan following, and is among the first video game characters to be successfully adapted to film. Lara Croft is also considered a sex symbol, one of the earliest in the industry to achieve widespread attention. The character's influence in the industry has been a point of contention among critics; viewpoints range from a positive agent of change in video games to a negative role model for young girls.
Thought: he Wikipedia page provides useful background information on the Lara Croft character and the Tomb Raider video game series she stars in, but does not directly answer which company developed the games. To find that, I will need to search more specifically for the developer.
Action: search-wikipedia
Action Input: Tomb Raider video game developer
Observation: Page: Tomb Raider Summary: Tomb Raider, known as Lara Croft: Tomb Raider from 2001 to 2008, is a media franchise that originated with an action-adventure video game series created by British video game developer Core Design. The franchise is currently owned by CDE Entertainment; it was formerly owned by Eidos Interactive, then by Square Enix Europe after Square Enix's acquisition of Eidos in 2009 until Embracer Group purchased the intellectual property alongside Eidos in 2022. The franchise focuses on the fictional British archaeologist Lara Croft, who travels around the world searching for lost artefacts and infiltrating dangerous tombs and ruins. Gameplay generally focuses on exploration, solving puzzles, navigating hostile environments filled with traps, and fighting enemies. Additional media has been developed for the franchise in the form of film adaptations, comics and novels. Development of the first Tomb Raider began in 1994; it was released two years later. Its critical and commercial success prompted Core Design to develop a new game annually for the next four years, which put a strain on staff. The sixth game, Tomb Raider: The Angel of Darkness, faced difficulties during development and was considered a failure at release. This prompted Eidos to switch development duties to Crystal Dynamics, which has been the series' primary developer since. Other developers have contributed to spin-off titles and ports of mainline entries. Tomb Raider games have sold over 95 million copies worldwide by 2022. while the entire franchise generated close to $1.2 billion in revenue by 2002. The series has received generally positive reviews from critics, and Lara Croft has become one of the most recognisable video game protagonists, winning accolades and earning places on the Walk of Game and Guinness World Records.
Thought: The Wikipedia page indicates that the original Tomb Raider video game series featuring Lara Croft was developed by the British video game company Core Design. Specifically, it states:

"Tomb Raider, known as Lara Croft: Tomb Raider from 2001 to 2008, is a media franchise that originated with an action-adventure video game series created by British video game developer Core Design."

And later confirms:

"Development of the first Tomb Raider began in 1994; it was released two years later."

So the company that originally developed the video games where Alice David voiced Lara Croft was Core Design.
Final Answer: The company that developed the video game series where Alice David voiced Lara Croft is Core Design.

Reflection: I assumed that the game was developed by Core Design, but I should have looked further into the observation to see that the game was later developed by Crystal Dynamics. I should have searched Crystal Dynamics to find out that they developed the game, instead of assuming that Core Design was the only developer.
</example>"""

self_reflection_prompt_template = """You are an advanced reasoning agent that can improve based on self-reflection. 
You will be given previous reasoning trial in which you were given access to tools: [{tool_names}] and a question to answer. 
You were unsuccessful in answering the question either because you guessed the wrong answer with Final Answer, or you used up your set number of reasoning sentences.
In a few sentences, diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure.
Use complete sentences.

You have access to below tool descriptions:
<tool_descriptions>
{tool_descriptions}
</tool_descriptions>

You have access to examples:
<examples>
{examples}
</examples>

Skip the preamble and return only the reflections - do not explain.
Previous trial:
Question: {question}
{scratchpad}

Reflection: """

def create_self_reflect_prompt(
        tools: Sequence[BaseTool],
        examples: str = self_reflect_examples,
        template: str = self_reflection_prompt_template,
    ) -> PromptTemplate:
    
        human_prompt = PromptTemplate(
            input_variables=[
                "question",
            ],
            partial_variables={
                "tool_names": ", ".join([tool.name for tool in tools]),
                "tool_descriptions": "\n".join(
                    [f"    {tool.name}: {tool.description}" for tool in tools]
                ),
                "examples": self_reflect_examples
            },
            template=template,
        )
        human_message_prompt = HumanMessagePromptTemplate(prompt=human_prompt)
        return ChatPromptTemplate.from_messages([human_message_prompt])
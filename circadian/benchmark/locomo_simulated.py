"""
Simulated LoCoMo-style benchmark conversations.

LoCoMo (ACL 2024): 10 conversations, 1,813 questions about:
- Disclosed facts (memory accuracy test)
- Adversarial questions (hallucination resistance — should NOT answer)

Each ConversationScenario has:
  - turns: list of (user_msg, agent_response) tuples
  - disclosed_facts: ground-truth facts the user mentioned
  - qa_pairs: (question, correct_answer, is_adversarial) tuples
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class QAPair:
    question: str
    correct_answer: str
    is_adversarial: bool  # True = user NEVER disclosed this fact


@dataclass
class ConversationScenario:
    scenario_id: str
    persona: str  # Brief description of the user persona
    turns: List[Tuple[str, str]]  # (user_message, agent_response)
    qa_pairs: List[QAPair]


SCENARIOS: List[ConversationScenario] = [
    ConversationScenario(
        scenario_id="tech_preferences",
        persona="Software developer with specific tool preferences",
        turns=[
            ("Hi, I've been using Claude Code for my React projects lately.",
             "That's great! Claude Code works well with React. What kind of projects are you building?"),
            ("Mostly dashboard UIs with TypeScript. I switched from VSCode because of the AI features.",
             "Nice! The AI-assisted coding makes a big difference. Do you use any particular frameworks?"),
            ("I use Next.js for most things now. Much faster than CRA.",
             "Next.js is solid for React SSR. Have you tried App Router yet?"),
            ("Yeah I'm on the App Router beta. It's pretty good but the learning curve is steep.",
             "I hear that. The server components take some getting used to."),
            ("By the way, I actually prefer Vim keybindings for editing.",
             "Classic! Modal editing is really powerful once you're comfortable with it."),
        ],
        qa_pairs=[
            # Disclosed facts — should be answerable
            QAPair("What frontend framework does the user prefer?", "Next.js", False),
            QAPair("What language does the user code in?", "TypeScript", False),
            QAPair("What code editor does the user use?", "Claude Code", False),
            QAPair("What keybindings does the user prefer?", "Vim keybindings", False),
            QAPair("What did the user switch from?", "VSCode", False),
            QAPair("What kind of projects does the user build?", "Dashboard UIs with TypeScript", False),
            QAPair("What router does the user use?", "App Router (Next.js)", False),
            # Adversarial — user NEVER said these
            QAPair("What IDE does the user prefer?", "[UNDISCLOSED]", True),
            QAPair("Does the user like Vim more than Emacs?", "[UNDISCLOSED]", True),
            QAPair("What operating system does the user run?", "[UNDISCLOSED]", True),
            QAPair("Does the user use JetBrains products?", "[UNDISCLOSED]", True),
            QAPair("What is the user's favorite color?", "[UNDISCLOSED]", True),
        ],
    ),
    ConversationScenario(
        scenario_id="dietary_preferences",
        persona="Person with specific dietary restrictions and meal preferences",
        turns=[
            ("I've been doing keto for about 6 months now.",
             "Keto can be really effective! How are you finding it?"),
            ("Pretty good, I lost 15 pounds. But the main reason I started was energy levels.",
             "That's a great motivation. Did you notice a big difference?"),
            ("Huge difference. I used to crash after lunch every day at work.",
             "That post-lunch crash is the worst. What kind of work do you do?"),
            ("I'm a financial analyst. Lots of spreadsheets and meetings.",
             "Sounds demanding. Do you meal prep on Sundays?"),
            ("Always! I make big batches of cauliflower rice bowls. So much easier.",
             "Cauliflower rice is great for keto. Do you add protein too?"),
            ("Grilled chicken or salmon usually. I try to hit 150g of protein daily.",
             "That's a solid protein target. Do you track everything in an app?"),
        ],
        qa_pairs=[
            # Disclosed
            QAPair("What diet is the user following?", "Keto", False),
            QAPair("How much weight did the user lose?", "15 pounds", False),
            QAPair("What is the user's occupation?", "Financial analyst", False),
            QAPair("What causes the user to crash?", "Post-lunch energy crashes", False),
            QAPair("What meal prep does the user make?", "Cauliflower rice bowls", False),
            QAPair("What protein sources does the user eat?", "Grilled chicken or salmon", False),
            QAPair("How much protein does the user aim for daily?", "150 grams", False),
            # Adversarial
            QAPair("Does the user eat red meat?", "[UNDISCLOSED]", True),
            QAPair("What snacks does the user eat on keto?", "[UNDISCLOSED]", True),
            QAPair("Does the user drink alcohol?", "[UNDISCLOSED]", True),
            QAPair("What is the user's height?", "[UNDISCLOSED]", True),
            QAPair("Does the user do intermittent fasting?", "[UNDISCLOSED]", True),
        ],
    ),
    ConversationScenario(
        scenario_id="travel_history",
        persona="Frequent traveler with strong opinions on destinations",
        turns=[
            ("I just got back from Japan, it was absolutely incredible.",
             "Amazing! What cities did you visit?"),
            ("Tokyo for 5 days, then Kyoto for 3. I could have spent a week in each.",
             "Classic itinerary! What was the highlight of Tokyo?"),
            ("The food honestly. Every corner had incredible ramen or yakitori.",
             "Japanese street food is on another level. Did you try any themed cafes?"),
            ("I went to amaid cafes in Akihabara. Very surreal experience lol.",
             "That sounds so cool! Were there any cultural things you did?"),
            ("lots of temple visits in Kyoto. Fushimi Inari was unreal at sunrise.",
             "The thousands of vermillion torii gates are iconic. Was it crowded?"),
            ("Surprisingly peaceful at 5am. I'd definitely recommend going early.",
             "Great tip! Any plans for your next trip?"),
            ("Probably Portugal. I've heard the seafood in Lisbon is phenomenal.",
             "Portugal is wonderful. The pastéis de nata alone are worth the trip!"),
        ],
        qa_pairs=[
            # Disclosed
            QAPair("What country did the user visit?", "Japan", False),
            QAPair("How long was the user in Tokyo?", "5 days", False),
            QAPair("How long was the user in Kyoto?", "3 days", False),
            QAPair("What did the user eat in Japan?", "Ramen and yakitori", False),
            QAPair("What neighborhood did the user visit for maid cafes?", "Akihabara", False),
            QAPair("What shrine did the user visit in Kyoto?", "Fushimi Inari", False),
            QAPair("What time did the user visit Fushimi Inari?", "5am", False),
            QAPair("Where does the user want to go next?", "Portugal (Lisbon)", False),
            # Adversarial
            QAPair("How much did the trip cost?", "[UNDISCLOSED]", True),
            QAPair("Did the user get a JR Pass?", "[UNDISCLOSED]", True),
            QAPair("What hotel did the user stay in?", "[UNDISCLOSED]", True),
            QAPair("Did the user visit Disneyland?", "[UNDISCLOSED]", True),
            QAPair("What language does the user speak?", "[UNDISCLOSED]", True),
        ],
    ),
    ConversationScenario(
        scenario_id="music_hobbies",
        persona="Music enthusiast with specific genre preferences and equipment",
        turns=[
            ("I've been getting into vinyl records lately. The sound quality is just different.",
             "Totally agree! There's something special about the warm analog sound."),
            ("Yeah exactly! I picked up a Technics SL-1200 from a pawn shop.",
             "A classic! Those are tanks. Did it need much work?"),
            ("Had to replace the cartridge but otherwise solid. Cost me $200.",
             "Great deal! What kind of music are you spinning?"),
            ("Mostly jazz and classical. Coltrane, Bill Evans, that kind of thing.",
             "Perfect for vinyl! Do you have a favorite pressing?"),
            ("Kind of Blue by Miles Davis. The Blue Note originals are gorgeous.",
             "Can't beat that. Have you tried any modern jazz too?"),
            ("最近Nujabesにはまっています。lo-fi hip hopとjazz rapの先がけです。",
             "Interesting pick! That's a unique blend of jazz and hip hop."),
            ("Yeah his sampling technique is incredible. Very different from Western stuff.",
             "Sampling philosophy varies a lot across cultures. Do you have other Japanese artists?"),
        ],
        qa_pairs=[
            # Disclosed
            QAPair("What format is the user collecting?", "Vinyl records", False),
            QAPair("What turntable does the user own?", "Technics SL-1200", False),
            QAPair("How much did the turntable cost?", "$200", False),
            QAPair("What genre of music does the user prefer?", "Jazz and classical", False),
            QAPair("What jazz artist does the user mention?", "Coltrane and Bill Evans", False),
            QAPair("What album is the user's favorite?", "Kind of Blue by Miles Davis (Blue Note)", False),
            QAPair("What Japanese artist does the user like?", "Nujabes", False),
            QAPair("What genre is Nujabes?", "Lo-fi hip hop and jazz rap", False),
            # Adversarial
            QAPair("Does the user play any instruments?", "[UNDISCLOSED]", True),
            QAPair("Does the user attend live concerts?", "[UNDISCLOSED]", True),
            QAPair("What country was the user's turntable made in?", "[UNDISCLOSED]", True),
            QAPair("Does the user use headphones or speakers?", "[UNDISCLOSED]", True),
        ],
    ),
    ConversationScenario(
        scenario_id="fitness_routine",
        persona="Fitness-focused person with a structured workout regimen",
        turns=[
            ("Hit a new PR on deadlift yesterday — 405 pounds!",
             "That's awesome! How long have you been training for?"),
            ("About 4 years consistently. But I took it seriously from the start.",
             "Great progress! Do you follow a specific program?"),
            ("I run nSuns. The volume is brutal but the gains are real.",
             "nSuns is solid for intermediate lifters. Do you do cardio too?"),
            ("Minimal. Maybe 20 minutes of rowing after lifting.",
             "That works. What about mobility work?"),
            ("I stretch every morning for 15 minutes. Hip flexors are tight from sitting.",
             "Smart. Sitting all day really does a number on the hips."),
            ("Yeah I'm at a desk 8+ hours. Trying to stand more but hard to remember.",
             "A standing desk could help! Do you track your calories?"),
            ("Yeah I use MacroFactor. Been eating 2800 calories to bulk.",
             "Solid approach. Are you doing a slow bulk or going aggressive?"),
        ],
        qa_pairs=[
            # Disclosed
            QAPair("What is the user's deadlift PR?", "405 pounds", False),
            QAPair("How long has the user been training?", "4 years", False),
            QAPair("What program does the user run?", "nSuns", False),
            QAPair("What cardio does the user do?", "20 minutes of rowing", False),
            QAPair("How long does the user stretch daily?", "15 minutes", False),
            QAPair("Why does the user have tight hip flexors?", "Sitting 8+ hours at desk", False),
            QAPair("What calorie tracking app does the user use?", "MacroFactor", False),
            QAPair("How many calories is the user eating?", "2800", False),
            QAPair("What is the user's goal?", "Bulking", False),
            # Adversarial
            QAPair("What is the user's weight?", "[UNDISCLOSED]", True),
            QAPair("Does the user take supplements?", "[UNDISCLOSED]", True),
            QAPair("What is the user's sleep schedule?", "[UNDISCLOSED]", True),
            QAPair("Does the user have a gym membership?", "[UNDISCLOSED]", True),
        ],
    ),
]

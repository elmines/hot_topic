import re

_CASE_INSENS_PATTS = [
    # Websites
    r"(dot|\.) ?(com|edu|org|net|fm)",

    # University ads
    r"rasmussen", # Too aggressive?
    r"Arizona State",

    # Coke Florida
    r"Coke Florida",
    r"Ice(-| )cold Coca(-| )Cola and football",
    r"championship combo",

    # Locality Name
    r"alachua",
    r"gainesville",

    # Local Businesses
    r"Spurrier's Grit(-| )?Iron Grill",
    r"(Original|All)( |-)American Kitchen",
    r"(V|B|Bee) Pizza",
    r"Chevrolet GMC",
    r"GMC Sierra",

    # Notional Ads
    r"T(-| )Mobile",
    r"T(-| )Satellite",
    r"Chevy Silverado",
    r"Burrito Factory",
    r"v(e|i)rbo", # too aggressive?
    r"MyFICO",
    r"Orderly Meds",
    r"Hollywood Feed",
    r"Amazon Hub Delivery",
    r"Support After Abortion",
    r"Ellie's Eden",
    r"Ellie Zedin",
    r"US Bank Business Essential",
    r"Alpha Insurance",
    r"Hartford",
    r"OnDeck",
    r"Grai?nger",
    r"American Express Business Gold Card",
    r"ACAS powers",
    r"Instagram Teen Accounts",
    r"Because your playbook ensures your arena is always ready for tip(-| )off",
    r"Offering the products you need all in one place",
    r"Azure Well",
    r"And with Fin, we've built the number one AI agent for customer service",

    # Some Amazon ad campaign
    r"Know a local business that will make a great partner",
    r"A local coffee shop owner, florist, automotive shop, dry cleaner, you name it",
    r"are sold near you",

    # American Express
    r"American Express",
    r"MX Business Gold Card",
    r"MXTravel",

    r"ACAST",
    r"packed with bold ingredients",
    r"Coligan",
    r"Advantage Business Program",
    r"Kraft Mac and Cheese",
    r"Weight Watchers",
    r"Celtic Bank",
    r"Simply Money",
    r"(C|k)alshi",
    r"AirMed",
    r"Good and the Beautiful's Reading",
    r"P(er|urr)fect Bistro",
    r"get everyone home safely",
    r"beat any price",
    r"sell more so you save more",
    r"you can get back to what matters most",
    r"epending on certain loan attributes",
    r"uilt for business",

    # VPN ads because everyone has them
    r"(Nord|Express) ?VPN",
    r"NordSteller",

    r"Worth Recruiting",
    r"Posit Connect",
    r"TalkPython",
    r"Squarespace",
    r"Delete ?Me",
    r"KRCS", # Some Apple retailer
    r"(?<!A)Udacity", # Udacity, not audacity
    r"SteamClock",
    r"Zbiotics",
    r"Jubilee Trading Co",
    r"Mint ?Mobile",
    r"Colligan",
    r"Uber Eats",
    r"Pepsi Prebiotic",
    r"Carvana",
    r"Ever ?Pass",
    r"Exclaimer",
    r"State Farm",
    r"Lowe'?s",
    r"Nespresso",
    r"Factor Meals",

    #Entertainment services--everything is called "plus" these days
    r"(MGM|Disney) ?(plus|\+)",
    r"Hulu",
    r"Britbox",
    r"RIPBOX", # mispronunciation of Britbox

    # Florida driver saftey
    r"isn't just a road, it's our workplace",
    r"Target Zero Initiative",
    r"This isn't just a road\.",
    r"It's our workplace\.",
    r"Slow down\.",
    r"Keep your distance\.",
    r"Stay focused\.",
    r"Your choices matter\.",
    r"Let's get everyone home safely\.",

    # Tobacco
    r"contains? nicotine",
    r"icotine is an addictive chemical",

    # Other public service campaigns
    r"opioid addiction is claiming lives",
    r"reminds you that April is Child Abuse Prevention Month",

    # Ads for other podcasts
    r"guests on my new podcast",
    r"(wherever|anywhere|everywhere) you (get|find) your podcast",

    # music (starting with a bracket or paren)
    r"[\[(](music|upbeat|singing|rock|sirens wailing)",
    r"♪",

    # General Ad Language
    r"brought to you by",
    r"is sponsored by"

]

_CASE_AD_PATTS = [
    r"American Airlines",
    r"US Bank", # Too short 
    r"Work in Progress",
    r"Topo Chico",
    r"Vanta", # Case-sensitive because it could be part of a larger word
    r"ASU",   # Same goes for an acronmy
    r"On Deck",
    r"Found",
    r"The Pizza", # Mis-transcription of "V Pizza"
    r"Alpha", # For alpha insurance
    r"Lem is",
    r"Burrito", # capitalized--probably ad for burrito factory

    # Short brand-names that probably need to be case-sensitive
    r"Shiny", # from Posit-Connect
    r"Sentry", # from TalkPython
    r"Seer", # from TalkPython
    r"Gusto",
    r"Factor",
    r"Oracle",
    r"Ring",
    r"Centuries",

    r"Zin", # nicotine patch
    r"Warning\.", # From tobacco ads
]


_WHITELIST = [
    r"@rasmussen_pole", # Is @ reserved in regex syntax?
    r"Slow music\.",
    r"Zeldin",
    r"zeldin",
]

AD_PATT = re.compile("(?i:" + "|".join(_CASE_INSENS_PATTS) + ")" + "|" + "|".join(_CASE_AD_PATTS))

__ALL__ = ["AD_PATT"]

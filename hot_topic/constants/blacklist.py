import re

_CASE_INSENS_PATTS = [
    # Websites
    r"(dot|\.)(-| )?(com|edu|org|net|fm|au|me|io|us)",
    # Toll-free numbers (still a thing apparently)
    r"1-800",
    r"one(-| )eight(-| )hundred",

    # University ads
    r"Rasmussen",
    r"Arizona ?State",
    r"Gortney ?Institute",
    r"Ottawa ?University",
    r"Western ?Governor'?s ?University",
    r"fully online master", # a lot of schools advertise this

    # Coke Florida
    r"Coke ?Florida",
    r"Ice(-| )cold Coca(-| )Cola and football",
    r"championship combo",

    # Locality Name
    r"alachua",
    r"gainesville",
    r"tuscaloosa",

    # Local Businesses
    r"Spur{1,2}ier'?s ?Gri(t|d)(-| )?Iron ?Grill",
    r"(Original|All)(-| )?American ?Kitchen",
    r"(V|B|Bee)(-| )Pizza",
    r"Chevrolet GMC",
    r"GMC Sierra",

    # National Ads
    r"T(-| )?(Mobile|Satellite)",
    r"Chevy Silverado",
    r"Burrito Factory",
    r"v(e|i)rbo", # too aggressive?
    r"My ?FICO",
    r"Orderly ?Meds",
    r"Hollywood ?Feed",
    r"Amazon ?Hub ?Delivery",
    r"Support ?After ?Abortion",
    r"Ellie's ?Eden",
    r"Ellie ?Zedin",
    r"US ?Bank ?Business ?Essential",
    r"Alpha ?Insurance",
    r"Hartford",
    r"OnDeck",
    r"Grai?nger",
    r"American ?Express ?Business ?Gold ?Card",
    r"ACAS powers",
    r"Instagram ?Teen ?Accounts",
    r"Because your playbook ensures your arena is always ready for tip(-| )off",
    r"Offering the products you need all in one place",
    r"Azure ?Well",
    r"And with Fin, we've built the number one AI agent for customer service",
    r"Tex Heart ?stone",
    r"Comcast",
    r"My ?Figs",

    # Legoland
    r"Lego ?Galaxy",
    r"Lego ?land",
    r"Galact(i|a) ?Coaster",


    r"Net ?Credit",
    r"Wendy'?s",
    r"Turbo ?Tax",
    r"Zaxby'?s",
    # Thrive Causemetics
    r"Thrive ?Cosmetics",
    r"Causemetics",
    r"Go ?Fund ?Me",

    # Insurance
    r"Geico",
    r"Progressive Casualty",
    r"Florida ?Blue?",

    r"Camellia",
    r"Blue ?Plate ? Mayo",
    r"Ralph'?s ?on ?the ?Park",
    r"Patriot ?Mobile",
    r"Washable ?Sofa",
    r"Pure ?Talk",
    r"Life ?Lock",
    r"Legacy ?Box",
    r"Get ?Sunday",
    r"Crocket{1,2} ?Coffee",
    r"Meaningful ?Beauty",
    r"Spin ?Quest",
    r"Modo ?Casino",
    r"Travel ?Texas",

    # Some Amazon ad campaign
    r"Know a local business that will make a great partner",
    r"A local coffee shop owner, florist, automotive shop, dry cleaner, you name it",
    r"refer a business today",
    r"wants to partner with your business",
    r"are sold near you",

    # American Express
    r"American ?Express",
    r"MX ?Business ?Gold ?Card",
    r"MXTravel",

    r"ACAST",
    r"packed with bold ingredients",
    r"Advantage ?Business ?Program",
    r"Kraft Mac and Cheese",
    r"Weight ?Watchers",
    r"Celtic ?Bank",
    r"Simply ?Money",

    # Kalshi
    r"(C|K)al ?sh(i|e)",
    r"trade on anything",

    r"Air ?Med",
    r"Good ?and ?the ?Beautiful'?s ?Reading",
    r"P(er|urr)fect ?Bistro",
    r"get everyone home safely",
    r"beat any price",
    r"sell more so you save more",
    r"you can get back to what matters most",
    r"epending on certain loan attributes",
    r"uilt for business",

    # VPN ads because everyone has them
    r"(Nord|Express) ?VPN",
    r"Nord ?Steller",

    r"Worth ?Recruiting",
    r"Posit ?Connect",
    r"Talk ?Python",
    r"Square ?Space",
    r"Delete ?Me",
    r"KRCS", # Some Apple retailer
    r"(?<!A)Udacity", # Udacity, not audacity
    r"Steam ?Clock",
    r"Z(ee)?(-| )?biotics",
    r"Jubilee ?Trading ?Co",
    r"Mint(-| )?Mobile",
    r"Col{1,2}igan",
    r"Uber ?Eats",
    r"Pepsi",
    r"Carvana",
    r"Ever ?Pass",
    r"Exclaimer",
    r"State ?Farm",
    r"Lowe'?s",
    r"Nespres{1,2}o",
    r"Factor ?Meals",
    r"(Dr\.?|Doctor) (Urtka|Oetker)", # Dr. Oetker pizza
    r"Dua? ?lux", # Dulux
    r"Live ?Nation",
    r"Cullen ?Jewelry",
    r"Comm ?Bank ?Yellow",
    r"Py ?Charm",
    r"Auth0",
    r"Fit ?Bud",
    r"Ecamm", # Require both m's. Otherwise we catch words like 'became'
    r"Vitally",
    r"Investor'?s ?Podcast",
    r"Lib ?syn",
    r"Fire ?stone",
    r"Slack ?Bot",
    r"web ?healer",
    r"counsel{1,2}ing ?tutor",
    r"counsel{1,2}ing ?skills ?academy",
    r"Reverie ?Power ?bed",
    r"Host ?Financial",
    r"Shopify",
    r"Base ?Lane",
    r"Chumba ?Casino",
    r"Sprot{1,2} ?Money", # Sprott Money
    r"Black ?Out ?Coffee",
    r"Il ?Tut{1,2}o", # Il Tutto
    r"Shop ?Beam",
    r"Beam'?s ?Dream ?Powder",
    r"Stop ?Box",
    r"American ?Financing",
    r"Helix ?Sleep",
    r"Genu ?cel{1,2}",
    r"(1|Fir)st ?(F|Ph)orm", # 1st Phorm
    r"My ?Bookie",
    r"Cocoa ?Beach",
    r"Total ?Wine",

    # r"Ketone", # Too many diet-podcasters talking about this

    r"Ashley ?Madison",
    r"Buy ?Ray ?Con",
    r"American ?Airlines",
    r"US ?Bank",
    r"Topo ?Chico",
    r"Colgate", # Colgate toothepaste ads
    r"Anabase", # Furnitue brand
    r"Neapolitan pizza",

    #Entertainment services--everything is called "plus" these days
    r"(MGM|Disney) ?(plus|\+)",
    r"Hulu",
    r"(Brit|Rip) ?Box", # BritBox
    r"(Z|X)umo ?Play",

    # Knoxville ads
    r"calling the curious",

    # Florida driver saftey
    r"isn't just a road, it's our workplace",
    r"Target ?Zero ?Initiative",
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
    r"iHeart",
    r"guests on my new podcast",
    r"listen to us (on|at)",
    r"subscribe and share",
    r"(wherever|anywhere|everywhere) you (get|find) your podcast",

    # music (starting with a bracket or paren)
    r"[\[(](music|upbeat|singing|rock|sirens wailing)",
    r"♪",

    # General Ad Language
    r"patreon",
    r"brought to you by",
    r"is sponsored by",
    r"off your order",
    r"with code",
    r"use (the )?code",
    r"at checkout",
    r"take advantage of",
    r"get started at",
    r"these offers end",
    r"exclusive offer",
    r"shop now at",
    r"full terms at",

]

_CASE_AD_PATTS = [
    r"Work in Progress",
    r"Vanta", # Case-sensitive because it could be part of a larger word
    r"ASU",   # Same goes for an acronmy
    r"On Deck",
    # r"Found", # Too-broad coverage in review
    r"The Pizza", # Mis-transcription of "V Pizza"
    r"Alpha", # For alpha insurance
    r"ALFA",
    r"ALPHA",

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
    r"Whole Foods", # Could be talking about "whole foods" (think MAHA movement)
    r"Progressive", # Casualty Insurance Company
    r"Steadily",    # Steadily Insurance

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

gaia_privacy_options = {
    "user_files": "Sensitive data extracted from user-provided local files or computed values (e.g., spreadsheets, PDFs, text files)",
    "names": "Personal names of people, first names, last names, full names",
    "emails": "Email addresses",
    "phones": "Phone numbers",
    "addresses": "Street addresses, cities, states, zip codes, countries",
    "identity_docs": "SSNs, passport numbers, driver license numbers, date of birth",
    "dates": "Specific dates, times, timestamps associated with user transactions",
    "organizations": "Company names, institution names, organization names",
    "locations": "Cities, countries, geographic locations",
    "numbers": "Specific numeric values, ID numbers, account numbers, quantities, measurements",
    "urls": "Website URLs and links",
    "search_results": "Sensitive data from web search results",
}

tau2_privacy_options = {
    "user_internal": "Internal user data retrieved from system tools: user IDs, account IDs, internal identifiers, and structured records returned by get_user_details, find_user_id_by_*, get_customer_by_* tools",
    "names": "Personal names of people, first names, last names, full names",
    "emails": "Email addresses",
    "phones": "Phone numbers",
    "addresses": "Street addresses, cities, states, zip codes, countries",
    "identity_docs": "SSNs, passport numbers, driver license numbers, dates of birth",
    "dates": "Specific dates, times, timestamps associated with user transactions",
    "payment_info": "Credit card numbers, payment method IDs, card details",
    "order_ids": "Order IDs, reservation IDs, ticket numbers, transaction IDs, confirmation numbers",
    "pricing": "Prices, amounts, totals, balances, fees, refund amounts",
    "products": "Publicly visible plan names, plan details, product types, product names, product descriptions, flight numbers, airport codes, flight schedules",
}

KEY_TO_CATEGORY = {
    "user_id": "user_internal", "account_id": "user_internal",
    "customer_id": "user_internal", "member_id": "user_internal",
    "name": "names", "first_name": "names", "last_name": "names", "full_name": "names",
    "email": "emails",
    "phone": "phones", "phone_number": "phones",
    "address": "addresses", "address1": "addresses", "address2": "addresses",
    "city": "addresses", "state": "addresses", "country": "addresses",
    "zip": "addresses", "zipcode": "addresses", "zip_code": "addresses",
    "dob": "identity_docs", "ssn": "identity_docs",
    "passport": "identity_docs", "driver_license": "identity_docs",
    "payment_method_id": "payment_info", "card_number": "payment_info",
    "payment_methods": "payment_info",
    "order_id": "order_ids", "reservation_id": "order_ids",
    "ticket_number": "order_ids", "transaction_id": "order_ids",
    "confirmation_number": "order_ids", "orders": "order_ids",
    "date": "dates", "time": "dates", "timestamp": "dates",
    "order_date": "dates", "created_at": "dates", "updated_at": "dates",
    "price": "pricing", "amount": "pricing", "total": "pricing",
    "balance": "pricing", "fee": "pricing", "total_amount": "pricing",
    "product_id": "products", "item_id": "products",
    "plan_id": "products", "device_id": "products",
    "iccid": "products", "imei": "products",
    "flight_number": "products", "passenger_id": "products",
}

KEY_TO_PREFIX = {
    "city": "CITY", "state": "STATE", "country": "COUNTRY",
    "zip": "ZIPCODE", "zipcode": "ZIPCODE", "zip_code": "ZIPCODE",
    "address": "ADDR", "address1": "ADDR", "address2": "ADDR",
    "email": "EMAIL",
    "name": "NAME", "first_name": "FIRST_NAME", "last_name": "LAST_NAME", "full_name": "FULL_NAME",
    "phone": "PHONE", "phone_number": "PHONE",
    "price": "PRICE", "amount": "PRICE", "total": "PRICE",
    "balance": "PRICE", "fee": "PRICE", "total_amount": "PRICE",
    "date": "DATE", "time": "DATE", "timestamp": "DATE",
    "order_date": "DATE", "created_at": "DATE", "updated_at": "DATE",
}

dataset_profiles = {
    "gaia": {
        "tools": [
            "web_search", 
            "visit_website", 
            "arxiv_search", 
            "wikipedia_lookup", 
            "python_exec", 
            "read_file", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["user_files"],
            2: ["user_files", "names", "emails", "phones", "addresses", "identity_docs", "dates", "organizations", "locations", "numbers"],
            3: ["user_files", "names", "emails", "phones", "addresses", "identity_docs", "dates", "organizations", "locations", "numbers", "urls", "search_results"],
        },
        "privacy_options": gaia_privacy_options
    },
    "tau2_airline": {
        "tools": [
            "book_reservation",
            "calculate",
            "cancel_reservation",
            "get_reservation_details",
            "get_user_details",
            "list_all_airports",
            "search_direct_flight",
            "search_onestop_flight",
            "send_certificate",
            "transfer_to_human_agents",
            "update_reservation_baggages",
            "update_reservation_flights",
            "update_reservation_passengers",
            "get_flight_status",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["user_internal"],
            2: ["user_internal", "names", "emails", "phones", "addresses", "identity_docs", "payment_info", "order_ids", "dates"],
            3: ["user_internal", "names", "emails", "phones", "addresses", "identity_docs", "payment_info", "order_ids", "dates", "pricing", "products"],
        },
        "privacy_options": tau2_privacy_options
    },
    "tau2_retail": {
        "tools": [
            "calculate",
            "cancel_pending_order",
            "exchange_delivered_order_items",
            "find_user_id_by_name_zip",
            "find_user_id_by_email",
            "get_order_details",
            "get_product_details",
            "get_item_details",
            "get_user_details",
            "list_all_product_types",
            "modify_pending_order_address",
            "modify_pending_order_items",
            "modify_pending_order_payment",
            "modify_user_address",
            "return_delivered_order_items",
            "transfer_to_human_agents",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["user_internal"],
            2: ["user_internal", "names", "emails", "phones", "addresses", "identity_docs", "payment_info", "order_ids", "dates"],
            3: ["user_internal", "names", "emails", "phones", "addresses", "identity_docs", "payment_info", "order_ids", "dates", "pricing", "products"],
        },
        "privacy_options": tau2_privacy_options
    },
    "gsm8k": {
        "tools": [
            "python_exec", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "math_qa": {
        "tools": [
            "python_exec", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "geometry3k": {
        "tools": [
            "python_exec",
            "read_file",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "mathvista": {
        "tools": [
            "python_exec",
            "read_file",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "scibench": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "sciq": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
    "truthful_qa": {
        "tools": [
            "web_search", 
            "visit_website", 
            "wikipedia_lookup", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
    "hotpot_qa": {
        "tools": [
            "web_search", 
            "visit_website", 
            "wikipedia_lookup", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
    "fever": {
        "tools": [
            "wikipedia_lookup", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
    "clutrr": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["names"],
        },
        "privacy_options": {
            "names": "Personal names of people, all names are enclosed in square brackets [] in the text"
        }
    },
    "agieval_lsat_ar": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["names", "numbers"],
        },
        "privacy_options": {
            "names": "All individual entity names or identifiers that represent participants, objects, or categories in the problem — including person names (e.g. Alice, Bob), single-letter codes (e.g. A, B, C), breed or species names (e.g. Labrador, Poodle), and Any other proper nouns serving as distinct entity labels",
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "med_qa": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["patient_profile"],
        },
        "privacy_options": {
            "patient_profile": "Overall patient background and clinical context, including simple demographic information (e.g., age group, sex), key past conditions, and the main current symptoms or exam findings needed to understand the question, without requiring precise identifiers or highly detailed lab values",
        }
    },
    "finqa": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "mmlu_professional_accounting": {
        "tools": [
            "python_exec",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "mmmu_accounting": {
        "tools": [
            "python_exec",
            "read_file",
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["numbers"],
        },
        "privacy_options": {
            "numbers": "Any numeric value (integers, decimals, fractions, percentages), quantities, counts, years, and measurements."
        }
    },
    "jeopardy_mc_history": {
        "tools": [
            "web_search", 
            "visit_website", 
            "wikipedia_lookup", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
    "jeopardy_mc_literature": {
        "tools": [
            "web_search", 
            "visit_website", 
            "wikipedia_lookup", 
            "final_answer"
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
    "image_generation": {
        "tools": [
            "ask_witness",
            "image_generation",
        ],
        "privacy_levels": {
            0: [],
            1: ["entities"],
        },
        "privacy_options": {
            "entities": "Every noun or noun phrase. This includes but is not limited to: animals (e.g. dogs, eagles, salmon), plants (e.g. oak trees, roses, bamboo), foods and beverages (e.g. pizza, green tea, chocolate), body parts and organs (e.g. liver, lungs, spine), objects and artifacts (e.g. telescope, compass, sword), buildings and landmarks (e.g. Eiffel Tower, Colosseum), songs and artworks (e.g. 'Mona Lisa', 'Bohemian Rhapsody'), books and documents (e.g. Magna Carta, Romeo and Juliet), cities and locations (e.g. Tokyo, Amazon Rainforest), people and characters (e.g. Cleopatra, Sherlock Holmes), natural phenomena (e.g. lightning, tides, rainbows), and ALL text appearing inside quotation marks."
        }
    },
}

def get_dataset_config(dataset_name):
    return dataset_profiles[dataset_name.lower()]

def has_privacy_level(dataset_name, level):
    dataset_name = dataset_name.lower()
    if dataset_name not in dataset_profiles:
        return False
    return level in dataset_profiles[dataset_name].get("privacy_levels", {})

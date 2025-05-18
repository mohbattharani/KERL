from SPARQLWrapper import SPARQLWrapper, JSON, RDF, N3
from pprint import pprint 
import urllib, json



def save_results (data, file_name):
    # Save the sample dictionary to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file, indent=4)

class FoodKG ():
    def __init__(self, url):
        # "http://128.213.11.13:9999/blazegraph/namespace/kb"
        self.sparql = SPARQLWrapper(url)
        self.tag_prefix = 'http://idea.rpi.edu/heals/kb/tag/'
        self.ingredient_prefix = 'http://idea.rpi.edu/heals/kb/ingredient/'
        self.micronutrients = [
            'recipe_calories',
            'recipe_cholesterol',
            'recipe_fat_cals',
            'recipe_fat_per_100g',
            'recipe_fiber',
            'recipe_protein',
            'recipe_salt_per_100g',
            'recipe_saturated_fat',
            'recipe_saturates_per_100g',
            'recipe_serving_size',
            'recipe_servings_per_recipe',
            'recipe_sodium',
            'recipe_sugar',
            'recipe_sugars_per_100g',
            'recipe_total_carbohydrates',
            'recipe_total_fat',
        ]   
        self.tags = ["1-day-or-more", "15-minutes-or-less", "3-steps-or-less", "30-minutes-or-less", "4-hours-or-less", "5-ingredients-or-less", "60-minutes-or-less", "african", "american", "appetizers", "apples", "asian", "asparagus", "australian", "bacon", "bananas", "bar-cookies", "barbecue", "bass", "bath-beauty", "beans", "beef", "beginner-cook", "belgian", "berries", "beverages", "bizarre", "black-beans", "blueberries", "bread-machine", "breads", "breakfast", "british-columbian", "broil", "brown-bag", "brownies", "brunch", "cajun", "cake-fillings-and-frostings", "cakes", "camping", "canadian", "candy", "canning", "carrots", "casseroles", "cheese", "chick-peas-garbanzos", "chicken", "chicken-breasts", "chinese", "chocolate", "chowders", "christmas", "citrus", "clams", "clear-soups", "cobblers-and-crisps", "cocktails", "coconut", "coffee-cakes", "college", "comfort-food", "condiments-etc", "cookies-and-brownies", "crab", "creole", "crock-pot-slow-cooker", "cuisine", "cupcakes", "curries", "desserts", "diabetic", "dinner-party", "dips", "drop-cookies", "easter", "easy", "egg-free", "eggs", "eggs-dairy", "elbow-macaroni", "english", "european", "fall", "filipino", "finger-food", "fish", "flat-shapes", "food-processor-blender", "for-1-or-2", "for-large-groups", "free-of-something", "freezer", "french", "freshwater-fish", "from-scratch", "fruit", "gifts", "gluten-free", "grains", "greek", "greens", "grilling", "ground-beef", "halibut", "ham", "hand-formed-cookies", "hawaiian", "healthy", "healthy-2", "heirloom-historical", "high-calcium", "high-in-something", "high-protein", "holiday-event", "hungarian", "independence-day", "indian", "inexpensive", "infant-baby-friendly", "iranian-persian", "italian", "jams-and-preserves", "japanese", "jewish-ashkenazi", "kid-friendly", "kosher", "lactose", "lasagna", "low-calorie", "low-carb", "low-cholesterol", "low-fat", "low-protein", "low-saturated-fat", "low-sodium", "lunch", "mahi-mahi", "main-dish", "mango", "marinades-and-rubs", "meat", "meatballs", "melons", "mexican", "microwave", "middle-eastern", "mixer", "muffins", "mushrooms", "new-years", "new-zealand", "no-cook", "non-food-products", "north-american", "northeastern-united-states", "novelty", "number-of-servings", "nuts", "oamc-freezer-make-ahead", "occasion", "omelets-and-frittatas", "one-dish-meal", "onions", "oranges", "oven", "pakistani", "pancakes-and-waffles", "papaya", "passover", "pasta", "pasta-rice-and-grains", "peanut-butter", "penne", "peppers", "picnic", "pies", "pies-and-tarts", "pineapple", "pizza", "pork", "pork-ribs", "pork-sausage", "potatoes", "potluck", "poultry", "presentation", "puddings-and-mousses", "punch", "quail", "quiche", "quick-breads", "refrigerator", "rice", "roast", "rolled-cookies", "rolls-biscuits", "romantic", "salad-dressings", "salads", "salmon", "salsas", "saltwater-fish", "sandwiches", "sauces", "savory", "savory-pies", "savory-sauces", "seafood", "seasonal", "served-cold", "served-hot", "shakes", "shellfish", "short-grain-rice", "shrimp", "side-dishes", "small-appliance", "smoker", "snacks", "soups-stews", "south-american", "south-west-pacific", "southern-united-states", "southwestern-united-states", "spaghetti", "spicy", "spinach", "spring", "squash", "steaks", "steam", "stir-fry", "stove-top", "strawberries", "stuffings-dressings", "summer", "superbowl", "sweet", "sweet-sauces", "taste-mood", "tex-mex", "thai", "thanksgiving", "to-go", "toddler-friendly", "tomatoes", "tropical-fruit", "tuna", "turkey", "valentines-day", "vegan", "vegetables", "vegetarian", "very-low-carbs", "wedding", "weeknight", "white-rice", "whole-chicken", "wild-game", "winter", "yams-sweet-potatoes", "yeast", "", "austrian", "baja", "baking", "beef-ribs", "birthday", "biscotti", "bisques-cream-soups", "brazilian", "broccoli", "brown-rice", "burgers", "californian", "caribbean", "central-american", "chard", "cheesecake", "cherries", "chicken-thighs-legs", "chili", "cinco-de-mayo", "collard-greens", "cooking-mixes", "copycat", "corn", "crusts-pastry-dough-2", "deep-fry", "duck", "dutch", "finnish", "frozen-desserts", "fudge", "garnishes", "gelatin", "german", "granola-and-porridge", "green-yellow-beans", "gumbo", "halloween", "herb-and-spice-mixes", "high-fiber", "household-cleansers", "irish", "jellies", "jewish-sephardi", "kiwifruit", "kwanzaa", "lamb-sheep", "leftovers", "lemon", "lentils", "lettuces", "lime", "long-grain-rice", "manicotti", "midwestern", "moroccan", "mothers-day", "mussels", "native-american", "no-shell-fish", "non-alcoholic", "oaxacan", "ontario", "oysters", "pacific-northwest", "pasta-shells", "peaches", "pears", "pitted-fruit", "plums", "polish", "pork-chops", "pork-loins", "portuguese", "pressure-cooker", "quebec", "raspberries", "roast-beef", "russian", "scandinavian", "scones", "scottish", "simply-potatoes", "smoothies", "south-african", "soy-tofu", "spanish", "spreads", "squid", "st-patricks-day", "steak", "stews", "stocks", "szechuan", "tarts", "tilapia", "turkey-breasts", "vietnamese", "water-bath", "whitefish", "whole-duck", "wings", "beijing", "cauliflower", "celebrity", "chilean", "chocolate-chip-cookies", "cod", "czech", "deer", "duck-breasts", "eggplant", "fillings-and-frostings-chocolate", "hanukkah", "heirloom-historical-recipes", "icelandic", "korean", "mashed-potatoes", "nepalese", "nut-free", "palestinian", "peruvian", "pet-food", "polynesian", "pot-pie", "puerto-rican", "pumpkin", "ravioli-tortellini", "rosh-hashana", "saudi-arabian", "scallops", "soul", "sugar-cookies", "swiss", "tempeh", "turkey-burgers", "veal", "welsh", "whole-turkey", "zucchini", "amish-mennonite", "beef-liver", "beef-organ-meats", "beef-sausage", "bok-choys", "crawfish", "dairy-free", "elk", "ethiopian", "grapes", "homeopathy-remedies", "ice-cream", "indonesian", "lebanese", "macaroni-and-cheese", "mardi-gras-carnival", "norwegian", "pennsylvania-dutch", "pressure-canning", "ragu-recipe-contest", "swedish", "trout", "turkish", "avocado", "bean-soup", "black-bean-soup", "cantonese", "chutneys", "cuban", "danish", "hunan", "laotian", "lobster", "meatloaf", "micro-melanesia", "mongolian", "onions-side-dishes", "orange-roughy", "rabbit", "ramadan", "super-bowl", "brewing", "catfish", "colombian", "dehydrator", "halloween-cakes", "iraqi", "medium-grain-rice", "moose", "sourdough", "argentine", "hidden-valley-ranch", "rosh-hashanah", "unprocessed-freezer", "cambodian", "chicken-livers", "labor-day", "libyan", "malaysian", "memorial-day", "pickeral", "georgian", "marinara-sauce", "sole-and-flounder", "spaghetti-sauce", "tomato-sauce", "tomatoes-sauces", "veggie-burgers", "artichoke", "chinese-new-year", "egyptian", "reynolds-wrap", "pheasant", "a1-sauce", "crock-pot-main-dish", "main-dish-beef", "venezuelan", "octopus", "slow-cooker", "sudanese", "angolan", "ecuadorean", "oatmeal", "fathers-day", "halloween-cupcakes", "namibian", "simply-potatoes2", "beef-kidney", "congolese", "nigerian", "dips-lunch-snacks", "costa-rican"]
    
    
    def fetch_all_tags(self):
        query = '''
        PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
        SELECT DISTINCT ?tag {
            ?r recipe-kb:tagged ?tag .
        }'''

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        tags = [x['tag']['value'] for x in results['results']['bindings']]
        return tags

    # Here, we extract all the dishes from KG (tagged or without tag)
    def fetch_all_dishes (self):
        query = '''
            PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
            SELECT DISTINCT ?r ?name {{
            ?r rdfs:label ?name .
            }}'''
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        dish_uris = [r['r']['value'] for r in results['results']['bindings']]
        dish_names = [r['name']['value'] for r in results['results']['bindings']]

        return dish_uris, dish_names
    
    def fetch_all_dishes_tagged (self):
        tags = self.fetch_all_tags()
        tagged_dishes = {}
        for t in tags:
            tagged_dishes[t] = self.get_dishes_for_tag(t)
        
        return tagged_dishes

    def get_name_dishuri(self, dish_uri):
        dish = '<{}>'.format(dish_uri)
        query = '''
                PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
                SELECT DISTINCT ?name {{
                {} rdfs:label ?name .
                }}'''.format(dish)
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        dishes = results['results']['bindings'][0]['name']['value']
        
        return dishes
    
    def get_uri_dishname(self, dish_name):
        query = '''
        PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
        SELECT DISTINCT ?recipe {{
            ?recipe rdfs:label ?name .
            FILTER (regex(?name, \"^{}$\", "i")) .
        }}
        '''.format(dish_name)
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        if 'results' in results and 'bindings' in results['results'] and len(results['results']['bindings']) > 0:
            recipe_uri = results['results']['bindings'][0]['recipe']['value']
            return recipe_uri
        else:
            return None
    
    # This function fetches all the dishes that have the given tag
    def get_dishes_for_tag(self, tag):
        tag = '<{}>'.format(tag)
        query = '''
            PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
            SELECT DISTINCT ?r ?name {{
                ?r recipe-kb:tagged {} .
                ?r rdfs:label ?name .
            }}'''.format(tag)

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        dishes = [(x['r']['value'], x['name']['value']) for x in results['results']['bindings']]
        
        return dishes
    
    def get_dishes_for_tag_name(self, tag_name):
        tag_uri = self.tag_prefix + tag_name
        dishes = self.get_dishes_for_tag(tag_uri)
        dishes = [dish[1] for dish in dishes]
        return dishes
    
    
    # This function fetches all the tags that have the given dish name
    def get_tags_for_dish_uri(self, dish_uri, remove_prefix = False):
        dish = '<{}>'.format(dish_uri)
        query = '''
        PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
        SELECT DISTINCT ?tag {{
            {} recipe-kb:tagged ?tag .
        }}
        '''.format(dish)
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        
        tags = [x['tag']['value'] for x in results['results']['bindings']]
        
        if (remove_prefix):
            tags = [t.replace(self.tag_prefix, "") for t in tags]
        
        return tags
    
    def get_tags_for_dish_name(self, dish_name):
        dish_uri = self.get_uri_dishname (dish_name)
        tags = self.get_tags_for_dish_uri (dish_uri)
        return tags


    # Given dish URI, we want to get all the ingredient used that the dish
    def get_ingredients_for_dish_uri(self, dish_uri):
        dish = '<{}>'.format(dish_uri)
        query = '''PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
            SELECT ?in {{
                {} recipe-kb:uses ?ii .
                ?ii recipe-kb:ing_name ?in
            }}
        '''.format(dish)

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        try:
            ingredients = [x['in']['value'] for x in results['results']['bindings']]
            ingredient_names = [urllib.parse.unquote(ing.split("/")[-1]) for ing in ingredients]
        except:
            ingredients = []
            ingredient_names = []
        
        return ingredients, ingredient_names

    # Given dish NAME, we want to get all the ingredient used that the dish

    def get_ingredients_for_dish_name(self, dish_name):
        uri = self.get_uri_dishname (dish_name)
        ingredients, ingredient_names = self.get_ingredients_for_dish_uri (uri)
        return ingredients, ingredient_names
    
    def select_recipes_contain_ingredients (self, ingredients, return_type = JSON):
        def generate_recipe_ingredient_query(ingredients):
            # Construct the FILTER clauses for each ingredient
            filter_clauses = f"regex(str(?ing_name), \"{ingredients[0]}\")"
            for ingredient in ingredients[1:]:
                filter_clauses += f"&&\n regex(str(?ing_name), \"{ingredient}\") "
            
            # Construct the complete SPARQL query
            query = f"""
            PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>

            SELECT DISTINCT ?name ?recipe
            WHERE {{
            ?recipe rdfs:label ?name .
            ?recipe recipe-kb:uses ?ing .
            ?ing recipe-kb:ing_name ?ing_name .
            FILTER ({filter_clauses})
            }}
            """

            return query
        
        query = generate_recipe_ingredient_query (ingredients)
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(return_type)
        results = self.sparql.query().convert()

        return results
    
    
    def get_nutritions_for_dish_name2(self,dish_name):

        # Construct the FILTER clause dynamically based on the attributes list
        filter_clause = ' || '.join([f'?attribute = recipe-kg:{attr}' for attr in self.micronutrients])
        
        query = f'''
        PREFIX recipe-kg: <http://idea.rpi.edu/heals/kb/>
        
        SELECT ?attribute ?value
        WHERE {{
        ?recipe rdfs:label ?name .
        ?recipe ?attribute ?value .
        FILTER ({filter_clause}) .
        FILTER (regex(?name, \"{dish_name}\")) .
        }}'''
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        data = results['results']['bindings']

        COLOR_CODE = {"green": 3, "orange": 2, "red": 1}
        
        result_dict = {}
        for d in data:
            try: 
                value = float (d['value']['value'])
            except:
                value = d['value']['value']
                if (value in COLOR_CODE):
                    value = COLOR_CODE[value]
                    
            result_dict[d['attribute']['value'].split("/")[-1]] = value
        
        return result_dict
    
    def get_nutritions_for_dish_name(self, dish_name):
        dish_uri = self.get_uri_dishname(dish_name)
        return self.get_nutritions_for_dish_uri(dish_uri)
    
    def get_nutritions_for_dish_uri(self, dish_uri):
        dish = '<{}>'.format(dish_uri)

        # Construct the FILTER clause dynamically based on the attributes list
        filter_clause = ' || '.join([f'?attribute = recipe-kg:{attr}' for attr in self.micronutrients])
        
        query = f'''
        PREFIX recipe-kg: <http://idea.rpi.edu/heals/kb/>
        
        SELECT ?attribute ?value
        WHERE {{
        {dish} ?attribute ?value .
        FILTER ({filter_clause}) .
        }}'''
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        data = results['results']['bindings']
        
        result_dict = {}
        for d in data:
            try: 
                value = float (d['value']['value'])
            except:
                value = d['value']['value']
            result_dict[d['attribute']['value'].split("/")[-1]] = value
        
        return result_dict
    
    def get_calories_for_dish_name(self, dish_name):
        dish_uri = self.get_uri_dishname(dish_name)
        return self.get_calories_for_dish_uri(dish_uri)
    
    def get_calories_for_dish_uri(self, dish_uri):
        dish = '<{}>'.format(dish_uri)
        query = f'''
        PREFIX recipe-kg: <http://idea.rpi.edu/heals/kb/>
        
        SELECT ?calories
        WHERE {{
        {dish} recipe-kg:recipe_calories ?calories .
        }}'''
        
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        data = results['results']['bindings']
        
        return float(data[0]['calories']['value'])
    
    def get_recipe_graph(self, dish_name):
        dish_uri = self.get_uri_dishname(dish_name) 
        ingredients = self.get_ingredients_for_dish_uri(dish_uri)[1]
        
        nutritions = self.get_nutritions_for_dish_uri(dish_uri)
        tags = [t.replace (self.tag_prefix, "") for t in self.get_tags_for_dish_uri(dish_uri)]    

        graph = {'title': dish_name}
        if (len(ingredients)):
            graph.update({'ingredients': ingredients})
        if (len(nutritions)):
            graph.update ({'nutritions': nutritions})
        if (len(tags)):
            graph.update ({'tags': tags})
        return graph
    
    
    def get_dishes_for_tag_and_ingredients(self, tag, ingredient_list):
        tag = '<{}>'.format(tag)
        ingredients_filter = ' '.join([
            '?r recipe-kb:uses ?ii{} . ?ii{} recipe-kb:ing_name <{}>'.format(i, i, urllib.parse.quote(ing)) 
            for i, ing in enumerate(ingredient_list)
        ])

        query = f'''
            PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
            SELECT DISTINCT ?r ?name {{
                ?r recipe-kb:tagged {tag} .
                ?r rdfs:label ?name .
                {ingredients_filter}
            }}
        '''

        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        dishes = [(x['r']['value'], x['name']['value']) for x in results['results']['bindings']]
        
        return dishes

    
    def generate_n_hop_query(self, node_uri, num_hops):
        if num_hops < 1:
            raise ValueError("Number of hops should be 1 or greater.")

        # Construct the initial part of the query
        query = f'''PREFIX recipe-kb: <http://idea.rpi.edu/heals/kb/>
        CONSTRUCT {{
        {node_uri} ?predicate1 ?neighbor1 .
        '''
        
        # Add lines for additional hops
        for hop in range(2, num_hops + 1):
            query += f'      ?neighbor{hop - 1} ?predicate{hop} ?neighbor{hop} .\n'

        # Complete the WHERE clause
        query += '    }\n'
        query += 'WHERE {{' + f'''
        {node_uri} ?predicate1 ?neighbor1 .
        '''
        
        # Add lines for additional hops in WHERE clause
        for hop in range(2, num_hops + 1):
            query += "OPTIONAL {" + f"?neighbor{hop - 1} ?predicate{hop} ?neighbor{hop}" +"}" +".\n" 
            #query += "" + f"?neighbor{hop - 1} ?predicate{hop} ?neighbor{hop}" +".\n" 

        query += '    }\n'

        # Complete the UNION clause
        
        query += f'''UNION {{
        ?neighbor1 ?predicate1 {node_uri} .
        '''
        
        # Add lines for additional hops in WHERE clause
        for hop in range(2, num_hops + 1):
            query += "" + f"?neighbor{hop - 1} ?predicate{hop} ?neighbor{hop}" +".\n" 

        query += '    }\n'
        query += '    }\n'
    
        return query 
    
    def construct_NHOP_graph (self, dish, N = 1, debug = False):

        dish = '<{}>'.format(dish)
        query = self.generate_n_hop_query(dish, N)
        
        if (debug):
            print (query)
        self.sparql.setQuery(query)
        self.sparql.setReturnFormat(RDF)
        graph = self.sparql.query().convert()

        return graph

    def generate_nhop_graph_triplets (self, dish, N = 1):
        
        graph = self.construct_NHOP_graph (dish, N)
        triplets = {}
        for s, p, o in graph:
            s = s.split("/")[-1].replace("%20", " ")
            p = p.split("/")[-1]
            o = o.split("/")[-1].replace("%20", " ")
            
            if (not s in triplets):
                triplets[s] = {}
            
            if (not p in triplets[s]):
                triplets[s][p] = []
            
            triplets[s][p].append (o)

        return triplets

    


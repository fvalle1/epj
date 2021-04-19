import inflect
import regex as re
inflect_engine = inflect.engine()

def singularize(word):
    singular = inflect_engine.singular_noun(word) #singularize
    if singular != False:
        return singular
    else:
        return word

def process_phrase(string: str):
    string = re.sub('&[a-zA-Z0-9]{1,}', '', string) #remove &aa
    string = re.sub('<[^<]+?>', '', string) #remove html
    string = re.sub('\([^<]+?\)', '', string) #parentesis
    string = re.sub('[-\.\:\\",!()?;_\'`¡ˆ]', '', string) #special characters
    string = re.sub('/', ' ', string) #replace / with space
    string = re.sub('[0-9]{1,}', '', string) # remove numbers
    string = re.sub('#[a-zA-Z0-9]{1,}', '', string) # remove hashtags
    string = re.sub('@[_a-zA-Z0-9]{1,}', '', string) # remove mentions
    string = re.sub('[ ]{2,}', '', string) # remove double spaces
    string = string.lower()
    string = ' '.join(list(map(singularize, string.split(" "))))
    return string
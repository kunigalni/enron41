
STEP 1 - Preprocessing 
1. Load Dataset and get all emails (takes abt a minute to load everything)
2. Apply a filter to only look at emails between Aug - Dec 2001
3. Remove signatures and replies 
4. Combine subject and body 
5. Remove duplicates 
6. Apply lemmatization (can verify by uncommenting the print statement)

STEP 2 - Keyword Frequency 
1. Count risk words 
2. TD IDF to find important words in dataset 
3. Count risk terms in each email 
4. Return 500 top emails with highest risk word frequency 
5. Return employees with highest risk word frequency 
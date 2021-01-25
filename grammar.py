

from collections import defaultdict
import random



# -- context-free grammar --

class C_F_G(object):
    def __init__(self):
        self.prod = defaultdict(list)

    def product(self, var1, var2):
        """ 
        :param var1: Old varaible will be replaced with new varaibles in production: var1 → var2, example: S → NP VP    
        :param var2: New varaibles in production which are separated by '|'. 
                    Each production is a sequence of symbols separated by whitespace.
       
        """
        prods = var2.split('|')
        for prod in prods:
            self.prod[var1].append(tuple(prod.split()))

    def rand_gen(self, symbol):
        """ 
        Generate a random sentence from the grammar by giving a symbol.
        """
        sentence = ''

        # select one production of this symbol randomly
        rand_prod = random.choice(self.prod[symbol])

        for sym in rand_prod:
            # for non-terminals, recurse
            if sym in self.prod:
                sentence += self.rand_gen(sym)
            else:
                sentence += sym + ' '

        return sentence
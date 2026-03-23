import re
from collections import Counter, OrderedDict
try:
    from torchtext.vocab import vocab as build_vocab
    _HAS_TORCHTEXT_V2_API = True
except ImportError:
    from torchtext.vocab import Vocab as build_vocab
    _HAS_TORCHTEXT_V2_API = False
from tqdm import tqdm

class TargetTokenizer:
    def __init__(self, df, special_symbols, UNK_IDX):
        self.sqamps = df['squared_amplitude'].tolist()
        
        # Keep only the patterns needed for the target sequence
        self.pattern_operators = {
            '+': re.compile(r'\+'), '-': re.compile(r'-'), '*': re.compile(r'\*'),
            ',': re.compile(r','), '^': re.compile(r'\^'), '%': re.compile(r'%'),
            '}': re.compile(r'\}'), '(': re.compile(r'\('), ')': re.compile(r'\)')
        }
        self.pattern_mass = re.compile(r'\b\w+_\w\b')
        self.pattern_s = re.compile(r'\b\w+_\d{2,}\b')
        self.pattern_reg_prop = re.compile(r'\b\w+_\d{1}\b')
        
        self.special_symbols = special_symbols
        self.UNK_IDX = UNK_IDX

    @staticmethod
    def remove_whitespace(expression):
        return re.sub(r'\s+', '', str(expression))

    @staticmethod
    def split_expression(expression):
        return re.split(r' ', expression)

    def tgt_tokenize(self, sqampl):
        """Tokenize target expression."""
        temp_sqampl = self.remove_whitespace(sqampl)
        
        for symbol, pattern in self.pattern_operators.items():
            temp_sqampl = pattern.sub(f' {symbol} ', temp_sqampl)
        
        for pattern in [self.pattern_reg_prop, self.pattern_mass, self.pattern_s]:
            temp_sqampl = pattern.sub(lambda match: f' {match.group(0)} ', temp_sqampl)
        
        temp_sqampl = re.sub(r' {2,}', ' ', temp_sqampl)
        return [token for token in self.split_expression(temp_sqampl) if token]

    def build_tgt_vocab(self):
        """Build vocabulary for target sequences."""
        counter = Counter()
        for eqn in tqdm(self.sqamps, desc='Processing target vocab'):
            counter.update(self.tgt_tokenize(eqn))
        if _HAS_TORCHTEXT_V2_API:
            voc = build_vocab(OrderedDict(counter), specials=self.special_symbols[:], special_first=True)
            voc.set_default_index(self.UNK_IDX)
        else:
            voc = build_vocab(counter, specials=self.special_symbols[:], specials_first=True)
        return voc
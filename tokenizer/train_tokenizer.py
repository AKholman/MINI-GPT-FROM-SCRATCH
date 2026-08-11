from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

# tokenizer trainer

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# Train tokenizer: here we start actual training the tokenizer

tokenizer.train(["data/tinystories.txt"], trainer)

# Test tokenizer

output = tokenizer.encode("Once upon a time there was a cat.")

print(output.tokens)
print(output.ids)

# Save tokenizer

tokenizer.save("tokenizer/tinystories.json")
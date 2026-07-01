"""
Sequence packing utilities to pack multiple short sequences into a single sequence block.
Prevents padding waste during LLM training.
"""

from typing import List

def pack_sequences(sequences: List[List[int]], max_length: int, eos_token_id: int = 2) -> List[List[int]]:
    """
    Pack short sequences separated by EOS tokens into sequence blocks of size max_length.
    """
    packed_blocks = []
    current_block = []
    
    for seq in sequences:
        # Include EOS token at the end of the sequence
        seq_with_eos = seq + [eos_token_id]
        
        while len(seq_with_eos) > 0:
            space_left = max_length - len(current_block)
            
            if len(seq_with_eos) <= space_left:
                current_block.extend(seq_with_eos)
                seq_with_eos = []
            else:
                current_block.extend(seq_with_eos[:space_left])
                packed_blocks.append(current_block)
                current_block = []
                seq_with_eos = seq_with_eos[space_left:]
                
            if len(current_block) == max_length:
                packed_blocks.append(current_block)
                current_block = []
                
    # If the last block is not full, we pad it
    if current_block:
        current_block.extend([0] * (max_length - len(current_block)))
        packed_blocks.append(current_block)
        
    return packed_blocks

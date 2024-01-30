def count_alphabet_occurrences(input_string):
    alphabet_counts = {}
    for char in input_string:
        if char.isalpha():
            char = char.lower()  #
            alphabet_counts[char] = alphabet_counts.get(char, 0) + 1
    return alphabet_counts
 
def find_max_occurrence(alphabet_counts):
    max_char = max(alphabet_counts, key=alphabet_counts.get)
    max_count = alphabet_counts[max_char]
    return max_char, max_count
 
if __name__ == "__main__":
    # Sinput string
    input_str = "hippopotamus"
 
    # Count alphabet 
    alphabet_counts = count_alphabet_occurrences(input_str)
 
    #  alphabet with the highest occurrence and count
    max_char, max_count = find_max_occurrence(alphabet_counts)
 
    # Print the result
    print(f"The maximally occurring character is '{max_char}' with occurrence count {max_count}.")
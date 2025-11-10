import pandas as pd
import json

# Set global variables
DATA_DIR = '../data/bible-kjv/'
PERICOPES_FILE = '../pericopes.csv'


def map_abbreviation_to_filename(abbreviation):
    """Map Bible book abbreviations to their corresponding JSON filenames."""
    abbrev_to_filename = {
        'GEN': 'Genesis.json',
        'EXO': 'Exodus.json',
        'LEV': 'Leviticus.json',
        'NUM': 'Numbers.json',
        'DEU': 'Deuteronomy.json',
        'JOS': 'Joshua.json',
        'JDG': 'Judges.json',
        'RUT': 'Ruth.json',
        '1SA': '1Samuel.json',
        '2SA': '2Samuel.json',
        '1KI': '1Kings.json',
        '2KI': '2Kings.json',
        '1CH': '1Chronicles.json',
        '2CH': '2Chronicles.json',
        'EZR': 'Ezra.json',
        'NEH': 'Nehemiah.json',
        'EST': 'Esther.json',
        'JOB': 'Job.json',
        'PSA': 'Psalms.json',
        'PRO': 'Proverbs.json',
        'ECC': 'Ecclesiastes.json',
        'SNG': 'SongofSolomon.json',
        'ISA': 'Isaiah.json',
        'JER': 'Jeremiah.json',
        'LAM': 'Lamentations.json',
        'EZK': 'Ezekiel.json',
        'DAN': 'Daniel.json',
        'HOS': 'Hosea.json',
        'JOL': 'Joel.json',
        'AMO': 'Amos.json',
        'OBA': 'Obadiah.json',
        'JON': 'Jonah.json',
        'MIC': 'Micah.json',
        'NAM': 'Nahum.json',
        'HAB': 'Habakkuk.json',
        'ZEP': 'Zephaniah.json',
        'HAG': 'Haggai.json',
        'ZEC': 'Zechariah.json',
        'MAL': 'Malachi.json',
        'MAT': 'Matthew.json',
        'MRK': 'Mark.json',
        'LUK': 'Luke.json',
        'JHN': 'John.json',
        'ACT': 'Acts.json',
        'ROM': 'Romans.json',
        '1CO': '1Corinthians.json',
        '2CO': '2Corinthians.json',
        'GAL': 'Galatians.json',
        'EPH': 'Ephesians.json',
        'PHP': 'Philippians.json',
        'COL': 'Colossians.json',
        '1TH': '1Thessalonians.json',
        '2TH': '2Thessalonians.json',
        '1TI': '1Timothy.json',
        '2TI': '2Timothy.json',
        'TIT': 'Titus.json',
        'PHM': 'Philemon.json',
        'HEB': 'Hebrews.json',
        'JAS': 'James.json',
        '1PE': '1Peter.json',
        '2PE': '2Peter.json',
        '1JN': '1John.json',
        '2JN': '2John.json',
        '3JN': '3John.json',
        'JUD': 'Jude.json',
        'REV': 'Revelation.json'
    }
    return abbrev_to_filename.get(abbreviation)

def get_passage_text(filename, start_chapter, start_verse, end_chapter, end_verse):
    """Retrieve the text of a passage from a JSON Bible file.

    Args:
        filename: JSON file name (e.g., '3John.json')
        start_chapter: Starting chapter number (int or str)
        start_verse: Starting verse number (int or str)
        end_chapter: Ending chapter number (int or str)
        end_verse: Ending verse number (int or str)

    Returns:
        str: The passage text with verses concatenated
    """
    with open(f"{DATA_DIR}{filename}", "r") as f:
        bible_data = json.load(f)

    # Convert to strings for comparison
    start_chapter = str(start_chapter)
    start_verse = str(start_verse)
    end_chapter = str(end_chapter)
    end_verse = str(end_verse)

    passage_text = []
    collecting = False

    for chapter_obj in bible_data["chapters"]:
        chapter_num = chapter_obj["chapter"]

        for verse_obj in chapter_obj["verses"]:
            verse_num = verse_obj["verse"]

            # Start collecting when we reach the start verse
            if chapter_num == start_chapter and verse_num == start_verse:
                collecting = True

            # Collect verse text if we're in the range
            if collecting:
                passage_text.append(verse_obj["text"])

            # Stop collecting after the end verse
            if chapter_num == end_chapter and verse_num == end_verse:
                collecting = False
                break

        if not collecting and passage_text:
            # We've already finished collecting
            break

    return " ".join(passage_text)

def main():
    # Load the pericopes data from a CSV file
    pericopes_df = pd.read_csv(PERICOPES_FILE)
    print(f"Loaded {len(pericopes_df)} pericopes.")

    # Map each pericope's book abbreviation to its JSON filename
    pericopes_df['filename'] = pericopes_df['Book'].apply(map_abbreviation_to_filename)

    # Print the updated DataFrame
    print(pericopes_df.head())

    # Get a list of all passages and text from pericopes
    passages = []
    for _, row in pericopes_df.iterrows():
        passage_text = get_passage_text(
            row['filename'],
            row['Chapter'],
            row['Start Verse'],
            row['Chapter'],
            row['End Verse']
        )
        passages.append(passage_text)

    # Print all collected passages
    for passage in passages[0:5]:
        print(passage)

    # Save the passages to a new CSV file
    pericopes_df['Passage Text'] = passages
    pericopes_df.to_csv('../data/pericopes_with_text.csv', index=False)

if __name__ == "__main__":
    main()
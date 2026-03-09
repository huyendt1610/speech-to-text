import os 

def get_libri_file_list(path_to_data_root = "C:/Users/HuyenDT/Downloads/LibriSpeech", 
                        include_splits = ["dev-clean"]):
    librispeech_data = []
    for s in include_splits: 
        path_to_split = os.path.join(path_to_data_root, s)  # format: speaker/section/audio
        
        for speaker in os.listdir(path_to_split): 
            path_to_speaker = os.path.join(path_to_split, speaker)
            # print(speaker)

            for section in os.listdir(path_to_speaker): 
                path_to_section = os.path.join(path_to_speaker, section)
                files = os.listdir(path_to_section)

                transcript_file = [path for path in files if ".txt" in path][0]
                with open(os.path.join(path_to_section, transcript_file), "r") as f: 
                    transcripts = f.readlines()

                for line in transcripts: 
                    split_line = line.split() # default is space => return an array
                    audio_root = split_line[0]
                    audio_file = audio_root + ".flac"
                    full_path_to_audio_file = os.path.join(path_to_section, audio_file)
                    transcript = " ".join(split_line[1:]).strip()

                    librispeech_data.append(
                        (full_path_to_audio_file, transcript)
                    )
    
    return librispeech_data

if __name__ == "__main__":
    l = get_libri_file_list()
    print(len(l))
    


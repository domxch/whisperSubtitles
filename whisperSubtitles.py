import glob
import os
import whisper
import whisper.utils


def format_timestamp(seconds: float, always_include_hours: bool = False, fractionalSeperator: str = '.'):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return f"{hours_marker}{minutes:02d}:{seconds:02d}{fractionalSeperator}{milliseconds:03d}"


model = whisper.load_model("large")
print("model loaded")
dir_name = '/content/'
# Get list of all files in a given directory sorted by name
list_of_files = sorted(filter(os.path.isfile,
                              glob.glob(dir_name + 'out*.mp3')))
# Iterate over sorted list of files and print the file paths
# one by one.
count = 0

for file_path in list_of_files:
    fileNumber = int(file_path[12:15])
    startingTimeMins = fileNumber * 2
    print(file_path + " " + str(fileNumber * 2))

    if startingTimeMins >= 0:
        text = model.transcribe(file_path, task='translate')
        print(str(type(text)))
        with open("/content/MVSD-390.srt", "a", encoding="utf-8") as srt_file:

            for i, segment in enumerate(text['segments'], start=1):
                print(segment)
                count = count + 1
                # write srt lines
                print(
                    f"{count}\n"
                    f"{format_timestamp(segment['start'] + (startingTimeMins * 60), always_include_hours=True, fractionalSeperator=',')} --> "
                    f"{format_timestamp(segment['end'] + (startingTimeMins * 60), always_include_hours=True, fractionalSeperator=',')}\n"
                    f"{segment['text']}\n",
                    file=srt_file,
                    flush=True,
                )
        print(text)
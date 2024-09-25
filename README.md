# music-xml-to-csv
Extract notes of each instrument and each voice from a xml file


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Installation

### Dependencies
```bash
pip install music21
```

### Installation Steps

```bash
git clone https://github.com/clarelinzx/music-xml-to-csv.git
```

## Usage
```python
from xml_parser import XmlParser
# Initialize the parser
parser = SymphonyParser('path/to/symphony.xml')
# Set file path
xml_path = os.path.join('put', 'your', 'xml', 'path', 'here', 'sheet.xml')
#  output directory default at /output/piece_name
output_dir = os.path.join('output', 'piece_name')
# Set options to control which information will be converted, or leave it blank (default all True)
options = (
    ('CONVERT_TEMPORAL_ATTR', True),
    ('CONVERT_NOTE', True),
    ('CONVERT_EXPRESSION', True),
)
# Set inst_with_multi_voice if there are any instrument or part that consists more than one voice 
# and you want to split them. 
inst_with_multi_voice = {
    # 'inst_name': (voice_num, start_idx)
    'horn12': (2, 1),
    'horn34': (2, 3),
    'trombone': (3, 1),
}
# Set allow_multi_pitch_all to control which instrument can play multiple notes at the same time (default False).
# Please make sure all instrument/track/voice name are in English; otherwise, you will need to define/modify the
# `allow_multi_pitch_all`(`ALLOW_MULTI_PITCH`) to set these attributes.
allow_multi_pitch_all = {
    'horn': False,
    'violin': True
}
# Parse note information
xml_parser = XmlParser(
    piece_name='piece_name',
    xml_path=xml_path,
    output_dir=output_dir,
    options=options,
    inst_with_multi_voice=inst_with_multi_voice,
    allow_multi_pitch_all=allow_multi_pitch_all
)
```

Here is an example:
```python
xml_path = os.path.join('xml_files', 'mozart_no41_mv1.xml')
output_dir = os.path.join('output', 'mozart_no41_mv1')
options = (
    ('CONVERT_TEMPORAL_ATTR', True),
    ('CONVERT_NOTE', True),
    ('CONVERT_EXPRESSION', True),
)
inst_with_multi_voice = {
        'horn12': (2,1),
        'horn34': (2,3),
        'trumpet12': (2,1),
    }
xml_parser = XmlParser(
    xml_path=xml_path,
    piece_name='mozart_no41_mv1',
    output_dir=output_dir,
    options = options,
    inst_with_multi_voice=inst_with_multi_voice
)
```

Command-line example:
```bash
# xml_path argument is required; piece_name and output_dir are optional
python xml_parser.py --xml_path ./xml_files/mozart_no41_mv1.xml
python xml_parser.py --xml_path ./xml_files/mozart_no41_mv1.xml \
--piece_name mozart_no41_mv1 \
--output_dir ./output/mozart_no41_mv1  
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

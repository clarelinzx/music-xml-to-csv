# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
# Name:         xml_parser.py
# Purpose:      Extract note information in xml (musicxml) to csv files
#
# Authors:      Zih-Syuan Lin, 庭康
# ------------------------------------------------------------------------------
import argparse
import logging
import json
import os
import re
import warnings

from fractions import Fraction
import numpy as np
import pandas as pd
from music21 import articulations, converter, dynamics, expressions, spanner, stream, note
from typing import List, Optional, Tuple, Union


# Configurations
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = '.'
EXPRESSIONS_CLASSES = [spanner.Slur, articulations, dynamics.Dynamic, expressions.TextExpression]
CSV_COLUMNS = {
    "downbeat": ['onset'],
    "time_signature": ['onset', 'measure', 'time_signature'],
    "note": [
        'onset', 'offset', 'midi_number', 'pitch_name', 'duration', 'staff', 
        'measure', 'beat_in_measure', 'instrument',
        'tie_type', 'mode', 'select_time'
    ],
    "expression": ['onset', 'offset', 'expression', 'instrument'], 
}
TEXT_EXPRESSION_ERROR_SET = {'[', ']', 'a 2', '♭', '♯', '[♭]', '[♯]'}
ALLOW_MULTI_PITCH_DEFAULT = False
ALLOW_MULTI_PITCH = {
    'piccolotrumpet': False,
    'trumpet': False,
    'horn': False,
    'english horn': False,
    'trombone': False,
    'basstrombone': False,
    'tuba': False,
    'piccolo': False,
    'flute': False,
    'oboe': False,
    'clarinet': False,
    'bassoon': False,
    'contrabassoon': False,
    'sopranosolo': False,
    'altochoir': False,
    'tenorchoir': False,
    'basschoir': False,
    'sopranochoir': False,
    'tenorsolo': False,
    'altosolo': False,
    'baritonesolo': False,
    'violin': True,
    'violin1': True,
    'violin2': True,
    'viola': True,
    'violoncello': True,
    'doublebass': True,
    'triangle': True,
    'cymbal': True,
    'timpani': True,
    'piano': True
}

def ensure_float(value):
    return float(Fraction(value))

def custom_warning_to_logging(message, category, filename, lineno, file=None, line=None):
        logging.warning(f'{filename}:{lineno}: {category.__name__}: {message}')

class XmlParser:
    def __init__(self, *,
        piece_name: str,
        xml_path: str,
        output_dir: Optional[str]=None,
        options: Optional[Union[dict, Tuple[Tuple, Tuple]]] = None,
        inst_with_multi_voice: dict = {},
        allow_multi_pitch_all: dict = ALLOW_MULTI_PITCH
    ):
        # Option setting
        self.options = {
            'CONVERT_TEMPORAL_ATTR': True,
            'CONVERT_NOTE': True,
            'CONVERT_EXPRESSION': True,
        }
        if options is not None:
            if isinstance(options, tuple):
                options = dict(tuple(options,))
            for k,v in dict(options).items():
                self.options.update({k:v})
        # set attributes
        self.piece_name = piece_name
        self.xml_path = xml_path
        if output_dir is None:
            self.output_dir = os.path.join(ROOT_DIR, 'output', self.piece_name)
        else:
            self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.inst_with_multi_voice = inst_with_multi_voice
        self.allow_multi_pitch_all = allow_multi_pitch_all

        self.info = {
            'measure_num': None,
            'time_signature': None
        }
        self.wait_to_stop_tie = []
        self.wrong_note_list = []
        log_file = os.path.join(self.output_dir, 'xml_parser.log')
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,  
            format="%(asctime)s - %(levelname)s - %(message)s", 
            datefmt="%Y-%m-%d %H:%M:%S",
            filemode='w'
        )
        warnings.showwarning = custom_warning_to_logging
        print("Load xml file...")
        logging.info("Load xml file...")
        self.xml_data = self.load_data(xml_path)
        self.full_score = self.transpose_pitch(self.xml_data)
        print(f"Create output directories under {self.output_dir}")
        logging.info(f"Create output directories under {self.output_dir}")
        self.create_output_dirs()
        print("Extract and save temporal attributes...")
        logging.info("Extract and save temporal attributes...")
        self.extract_temporal_attr()
        self.save_temporal_attr()
        print("Extract and save score info: note, expression...")
        logging.info("Extract and save score info: note, expression...")
        self.extract_and_save_score_info()
        print("Save info...")
        logging.info("Save info...")
        self.save_info()
        print("Revise note directory...")
        logging.info("Revise note directory...")
        self.revise_note_directory()
        print("Calculate note played for each instrument/voice...")
        logging.info("Calculate note played for each instrument/voice...")
        self.calculate_note_played()
        print("Done.")
        logging.info("Done.")

    def create_output_dirs(self) -> None:
        _dirs = [
            os.path.join(self.output_dir, file_name) for file_name in ['temporal_attributes', 'expression', 'note']
        ]
        for d in _dirs:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def load_data(xml_path):
        xml_data = converter.parse(xml_path)
        return xml_data

    @staticmethod
    def transpose_pitch(xml_data):
        logging.info("Transposing pitch to sounding pitch...")  # e.g. F horn, Bb clarinet
        return xml_data.toSoundingPitch()
    
    def save_csv(self, *, converting: str, data: List, file_name: str, columns: List[str]):
        '''Save csv file at "self.output_dir/converting/here.csv.'''
        df = pd.DataFrame(columns=columns, data=data)
        file_name = file_name[:-1] if file_name[-1] == '.' else file_name
        save_at = os.path.join(self.output_dir, converting, f"{file_name}.csv")
        logging.info(f"Saving {converting:<20} {file_name:<20}: {save_at:<80}")
        df.to_csv(save_at, index=False)

    def save_txt(self, *, converting: str, data: Union[List[str], str, int, float], file_name: str):
        save_at = os.path.join(self.output_dir, converting, f'{file_name}.txt')
        with open(save_at, 'w') as f:
            try:
                if isinstance(data, list):
                    for line in data:
                        f.write(line + '\n')
                else:
                    f.write(str(data))
            except TypeError as e:
                raise TypeError("data must be List[str], str, int, or float!!")

    def save_json(self, *, data, file_name: str):
        save_info_at=os.path.join(self.output_dir, f'{file_name}.json')
        with open(save_info_at, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    
    def save_info(self) -> None:
        # Save temporal attributes of this movement, including: measure num, time signature, wrong_note_list
        self.info.update({'wrong_note_list': self.wrong_note_list})
        self.save_json(data=self.info, file_name='temporal_attr')

    def extract_temporal_attr(self):
        if self.options['CONVERT_TEMPORAL_ATTR']:
            for part in self.full_score.iter().activeElementList:
                if 'Part' in part.classes and part.isStream:
                    break
            xf = part.flatten()
            measure_num = len(part.getElementsByClass(stream.Measure))
            # record all time signature changed in the score: (offset, measure, time signature)
            ts_list = []  
            for ts in xf.getTimeSignatures():
                ts_list.append((ts.offset, ts.measureNumber, ts.barDuration.quarterLength))
            downbeat_list = []
            for i in range(len(ts_list)-1):
                downbeat_list.extend(np.arange(ts_list[i][0], ts_list[i+1][0], ts_list[i][2]))
            downbeat_list.extend(np.arange(ts_list[-1][0], ts_list[-1][0]+ts_list[-1][2] * (measure_num - ts_list[-1][1])+1, ts_list[-1][2]))
            self.downbeat_list = downbeat_list
            self.info.update({
                'measure_num': measure_num,
                'time_signature': ts_list,
            })
        else:
            logging.info(f"CONVERT_TEMPORAL_ATTR is {self.options['CONVERT_TEMPORAL_ATTR']}: did not convert downbeat information.")
            self.downbeat_list = None
    
    def save_temporal_attr(self):
        if self.options['CONVERT_TEMPORAL_ATTR']:
            self.save_csv(
                converting='temporal_attributes',
                data=self.downbeat_list,
                file_name="downbeat",
                columns=CSV_COLUMNS['downbeat'],
            )
            self.save_csv(
                converting='temporal_attributes',
                data=self.info['time_signature'],
                file_name="time_signature",
                columns=CSV_COLUMNS['time_signature'],
            )
            self.save_txt(
                converting='temporal_attributes',
                data=self.info['measure_num'],
                file_name='measure_num',
            )

    def extract_and_save_score_info(self):
        if not self.options['CONVERT_NOTE'] and not self.options['CONVERT_EXPRESSION']:
            return
        self.inst_new_name_dict = {}
        self.all_part = []
        self.all_inst = []
        for part in self.full_score.iter().activeElementList:
            copy_flag = False
            self.all_part.append(part)
            if part.isStream and 'Part' in part.classes:
                inst_name = part.getInstrument().bestName().replace('/', '')  # change here
                if inst_name in self.inst_with_multi_voice.keys():
                    voice_num, start_idx = self.inst_with_multi_voice[inst_name]
                    many_parts = list(part.voicesToParts())
                    if len(many_parts) != voice_num:
                        if len(many_parts)==1:
                            copy_flag = True
                            warnings.warn(f"COPY CONVERTED FILE(S): Instrument(track) {inst_name} has different voice_num according to config.py, len()={len(many_parts)} != {voice_num}.", UserWarning)
                            print("WARNING")
                        else:
                            raise ValueError(f"FAILED TO CONVERT: Instrument(track) {inst_name} has different voice_num according to config.py, len()={len(many_parts)} != {voice_num}") 
                    for edx, each_part in enumerate(many_parts, start=start_idx):
                        new_inst_name = re.sub("\d+", "", inst_name) + str(edx)
                        expr_list = self.extract_expression_info(each_part, new_inst_name)
                        note_list = self.extract_note_info(each_part, new_inst_name)
                        self.save_score_info(inst_name=new_inst_name, expr_list=expr_list, note_list=note_list)
                    if copy_flag:
                        for edx in range(start_idx+1, start_idx+voice_num):
                            new_inst_name = re.sub("\d+", "", inst_name) + str(edx)
                            self.save_score_info(inst_name=new_inst_name, expr_list=expr_list, note_list=note_list)
                else:
                    expr_list = self.extract_expression_info(part, inst_name)
                    note_list = self.extract_note_info(part, inst_name)
                    self.save_score_info(inst_name=inst_name, expr_list=expr_list, note_list=note_list)

    def save_score_info(self, *, inst_name, expr_list, note_list):
        if self.options['CONVERT_NOTE']:
            self.save_csv(
                converting='note',
                data=note_list,
                file_name=inst_name,
                columns=CSV_COLUMNS['note']
            )
        if self.options['CONVERT_EXPRESSION']:
            self.save_csv(
                converting='expression',
                data=expr_list,
                file_name=inst_name,
                columns=CSV_COLUMNS['expression']
            )

    def extract_expression_info(self, part, inst_name):
        def check_slur_valid(obj):
            return True if len(obj)==2 else False

        if not self.options['CONVERT_EXPRESSION']:
            return None
        
        expr_list = []  
        part = part.flatten()
        for expr in EXPRESSIONS_CLASSES:
            if expr == articulations:
                for n in part.notes.stream().iter():
                    if len(n.articulations) > 0:
                        for arc in n.articulations:
                            expr_list.append([n.offset, n.offset,  arc.name, inst_name])
            else:
                found = part.getElementsByClass(expr).stream()
                for f in found.iter():
                    if expr == spanner.Slur:
                        if check_slur_valid(f):
                            expr_list.append([
                                f[0].getOffsetInHierarchy(self.full_score), f[1].getOffsetInHierarchy(self.full_score), 'legato', inst_name]
                            )
                    elif expr == spanner.Line:
                        expr_list.append((
                            f[0].getOffsetInHierarchy(part), f[0].getOffsetInHierarchy(part), 'accent', inst_name))
                    elif expr == articulations.Spiccato:
                        pass
                    elif expr == dynamics.Dynamic:
                        expr_list.append([
                            f.getOffsetInHierarchy(part), f.getOffsetInHierarchy(part), f.value, inst_name])
                    elif expr == expressions.TextExpression:
                        if f.content not in TEXT_EXPRESSION_ERROR_SET:
                            expr_list.append([
                                f.getOffsetInHierarchy(part), f.getOffsetInHierarchy(part), f.content, inst_name])
                    else:
                        raise Exception("Sorry, No this type of expression.")
        return expr_list

    def extract_note_info(self, part, inst_name):
        if not self.options['CONVERT_NOTE']:
            return None
        
        self.wait_to_stop_tie.clear()  # clean the waiting list
        note_list = []
        for element in part.flatten().notes:
            if element.isChord:
                for a_note in element.notes:
                    a_note_list = self.note_xml_parse(
                        mode='chord', a_note=a_note, a_chord=element, inst_name=inst_name
                    )
                    if a_note_list is not None:  # if it is not a waiting note (continue, holding)
                        note_list.append(a_note_list)
            elif element.isNote or isinstance(element, note.Unpitched):
                a_note_list = self.note_xml_parse(
                    mode='note', a_note=element, inst_name=inst_name
                )
                if a_note_list is not None:
                    note_list.append(a_note_list)
        # add waiting to stop tie list
        if self.wait_to_stop_tie:
            note_list += [_.convert_to_a_note_list()[:-1] + ['in_waiting_list'] for _ in self.wait_to_stop_tie]
            warning_str = f"[Queue Error] Until song ended, some note still holding: len(wait_to_stop_tie) = {len(self.wait_to_stop_tie)}. Add them into note_list."
            warnings.warn(warning_str, UserWarning)

        return note_list

    def note_xml_parse(self, *, mode, a_note, inst_name, a_chord=None) -> List[str]:
        # Default staff
        staff = 0
        # Attributes from a_note: onset, pitch_name, midi_number, duration
        if isinstance(a_note, note.Unpitched):  # Default midi_number to 0 if pitch is None
            pitch_name = "unpitched"
            midi_number = a_note.pitch.midi if hasattr(a_note, 'pitch') and a_note.pitch is not None else 0
        else:
            pitch_name = a_note.pitch.nameWithOctave
            midi_number = a_note.pitch.midi
        # Attributes from a_note or a_chord: measure, beat_in_measure
        if mode == 'note':
            duration = a_note.duration.quarterLength
            onset = a_note.offset
            measure = a_note.measureNumber
            beat_in_measure = a_note.beat
            note_inst_name = a_note.getInstrument().bestName().replace('/', '')
        elif mode == 'chord':
            duration = a_chord.duration.quarterLength
            onset = a_chord.offset
            measure = a_chord.measureNumber
            beat_in_measure = a_chord.beat
            note_inst_name = a_chord.getInstrument().bestName().replace('/', '')
        if re.sub("\d+", "", note_inst_name) != re.sub("\d+", "", inst_name):
            warnings.warn(f'part and note has different inst_name: part {inst_name} vs note {note_inst_name}',UserWarning)
        # Return
        select_time = 0
        a_note_list = [
            ensure_float(onset), ensure_float(onset)+ensure_float(duration), int(midi_number), pitch_name, 
            ensure_float(duration), staff, int(measure), ensure_float(beat_in_measure), inst_name, a_note.tie, mode, select_time
            ]
        return self.revise_a_note_list(a_note, a_note_list)
        
    def revise_a_note_list(self, a_note, a_note_list):
        # if a_note is not tie
        if isinstance(a_note, note.Unpitched) or (a_note.tie is None):
            return [str(_) for _ in a_note_list]
        # if a_note is tie
        elif a_note.tie is not None:
            # if a_note is tie.start
            if a_note.tie.type == 'start':
                self.wait_to_stop_tie.append(WaitNote(a_note_list))
                return None  # because this note is still waiting
            # if a_note is tie.continue
            elif a_note.tie.type == 'continue':
                selected_note = self.select_note_for_tie(a_note_list)
                if selected_note == 'continue':
                    return None
                elif selected_note == 'no_match':
                    warnings.warn('cannot find match note. just add this note anyway.', UserWarning)
                    return [str(_) for _ in a_note_list[:-1]] + ['add_it_anyway']
                else:
                    raise('unexpected error in line376.')
            # if a_note is tie.stop
            elif a_note.tie.type == 'stop':
                selected_note = self.select_note_for_tie(a_note_list)
                if isinstance(selected_note, list):
                    if len(selected_note) != 12: raise ValueError("selected_note is list but in weird length.")
                    return [str(_) for _ in selected_note]
                elif selected_note == 'no_match':
                    warnings.warn('cannot find match note. just add this note anyway.', UserWarning)
                    return [str(_) for _ in a_note_list[:-1]] + ['add_it_anyway']
                else:
                    raise('unexpected error in line358.')
        else:
            raise ValueError(f"Fail to revise a_note {a_note} with {a_note_list}")
    
    def select_note_for_tie(self, current_note_list):
        current_onset, _, current_midi_number, _, current_duration, _, _, _, _, tie_type, _, _ = current_note_list
        for wdx, waiting_note in enumerate(self.wait_to_stop_tie):
            if waiting_note.midi_number == current_midi_number:
                if waiting_note.offset != current_onset:
                    warnings.warn(f"matching notes are not consecutive; continue find next. (current's {current_note_list}) (matched's {waiting_note.convert_to_a_note_list()})", UserWarning)
                    continue
                else:  # matched
                    waiting_note.select_time += 1
                    waiting_note.duration += current_duration
                    waiting_note.offset += current_duration
                if tie_type.type == 'continue':
                    waiting_note.tie_type.type = 'continue'
                    return 'continue'
                elif tie_type.type == 'stop':
                    selected_note = waiting_note.convert_to_a_note_list()
                    del self.wait_to_stop_tie[wdx]  # pop this item
                    return selected_note
                else:
                    raise ValueError("current note's tie_type is not continue or stop.")
            warnings.warn('cannot find a match note.', UserWarning)            
        if tie_type.type == 'continue':
            tie_type.type = 'start'
            self.wait_to_stop_tie.append(WaitNote(current_note_list[:9] + [tie_type] + current_note_list[10:]))
            warnings.warn("this 'continue' note cannot match any previos start/continue note. set it as a new start and add to waiting list.", UserWarning)
            return 'continue'
        else:
            return 'no_match'
    

    def revise_note_directory(self):
        new_note_path = os.path.join(self.output_dir, 'new_note')
        os.makedirs(new_note_path,exist_ok=True)
        for inst in os.listdir(os.path.join(self.output_dir,'note')):
            if inst.endswith('.csv'):
                note_df = pd.read_csv(
                    os.path.join(self.output_dir,'note', inst)
                )
                new_note_df = note_df[[
                    'onset', 'offset', 'midi_number', 'pitch_name', 'duration', 'staff', 
                    'measure', 'beat_in_measure','instrument']]
                new_note_df.set_index('onset', inplace=True)
                new_note_df.to_csv(
                    os.path.join(new_note_path, inst)
                )
        # remove old note directory and rename new_note -> note
        for root, dirs, files in os.walk(os.path.join(self.output_dir, 'note'), topdown=False):
            for file in files:
                os.remove(os.path.join(root, file)) 
            for d in dirs:
                os.rmdir(os.path.join(root, d))  
        os.rmdir(os.path.join(self.output_dir, 'note')) 
        os.rename(os.path.join(self.output_dir, 'new_note'), os.path.join(self.output_dir, 'note')) 

        
    def calculate_note_played(self):
        note_played_path = os.path.join(self.output_dir, 'note_played')
        os.makedirs(note_played_path,exist_ok=True)
        note_played = []
        for inst in os.listdir(
            os.path.join(self.output_dir,'note')
        ):
            if inst.endswith('.csv'):
                note_df = pd.read_csv(
                    os.path.join(self.output_dir, 'note', inst)
                )
                note_played.append([inst[:-4], note_df.shape[0]])
        note_played = pd.DataFrame(note_played, columns=['instrument_name', 'n_note_played'])
        note_played = note_played.sort_values('instrument_name')
        note_played.to_csv(os.path.join(note_played_path, 'note_played.csv'))
        

class WaitNote:
    def __init__(self, a_note_list):
        self.onset, self.offset, self.midi_number, self.pitch_name, self.duration, self.staff, self.measure, self.beat_in_measure, self.inst_name, self.tie_type, self.mode, self.select_time = a_note_list

    def convert_to_a_note_list(self):
        return [self.onset, self.offset, self.midi_number, self.pitch_name, self.duration, self.staff, self.measure, self.beat_in_measure, self.inst_name, self.tie_type, self.mode, self.select_time]
    
    def check_if_match(self, another_midi_number):
        return True if self.midi_number == another_midi_number else False
    

def main():
    parser = argparse.ArgumentParser(description='Validate dataset movements and generate piano roll visualizations.')
    parser.add_argument('--xml_path', type=str, required=True, help='Path to the XML file')
    parser.add_argument('--piece_name', type=str, required=False, default=None, help='Name of the piece')
    parser.add_argument('--output_dir', type=str, required=False, default=None, help='Directory to save output')
    args = parser.parse_args()
    xml_path = args.xml_path
    piece_name = args.piece_name
    output_dir = args.output_dir
    if piece_name is None:
        piece_name = os.path.splitext(os.path.basename(args.xml_path))[0]
    xml_parser = XmlParser(
        xml_path=xml_path,
        piece_name=piece_name,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()


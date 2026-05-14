
import React, {useState} from 'react'
import { StyleSheet, Text, View, StatusBar, Platform,
  Dimensions, useColorScheme, TouchableOpacity, ImageBackground,
  ScrollView
 } from 'react-native'

import axios from 'axios'
import {Audio} from 'expo-av'
import * as DocmentPicker from 'expo-document-picker'

const DarkLogo = require('../assets/images/background.jpeg')
const API_URL = 'http://172.20.10.10:8000/api/stt/predict';
export const {height, width} = Dimensions.get('window')
export const fonts = {Bold: {fontFamily: 'Roboto-Bold'}}

const App = () => {
  const isDarkMode = useColorScheme() === 'dark'
  const [recording, setRecording] = useState<Audio.Recording|null>(null);
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState(''); 
  const [duration, setDuration] = useState<number|null>(null) 
  const [label, setLabel] = useState(''); 
  const [language, setLanguage] = useState('au')
  const [modelName, setModelName] = useState('whisper')

  const backgrounStyle = {backgroundColor: isDarkMode? '#000': '#FFF'}

  const startRecording = async() => {
    const {granted} = await Audio.requestPermissionsAsync(); 
    if (!granted) return; 

    await Audio.setAudioModeAsync({allowsRecordingIOS: true, playsInSilentModeIOS: true})

    const rec = new Audio.Recording(); 
    await rec.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY)
    await rec.startAsync(); 
    setRecording(rec)
    setIsRecording(true)
    setLabel('Recording...')

  }

  const stopRecording = async () => {
    if (!recording) return
    await recording.stopAndUnloadAsync() 
    const uri = recording.getURI() 
    setRecording(null)
    setIsRecording(false)
    if (uri) sendToAPI(uri, 'recording.wav', 'audio/wav')
  }

  const pickAudioFile = async () => {
    const result = await DocmentPicker.getDocumentAsync({type: 'audio/*'})
    if (!result.canceled){
      const asset = result.assets[0]
      sendToAPI(asset.uri, asset.name, asset.mimeType?? 'audio/mpeg')
    }
  }

  const sendToAPI = async (uri: string, name: string, mimeType: string) => {
    setLabel('Processing...')
    setTranscript('')
    const formData = new FormData(); 
    formData.append('file', { uri, name, type: mimeType} as any)
    formData.append('la', language)
    formData.append('model_name', modelName)
    try{
      const res = await axios.post(API_URL, formData, {
        headers: {'Content-Type': 'multipart/form-data'}
      })

      setTranscript(res.data.transcript) 
      setDuration(res.data.duration)
      setLabel('')
    }
    catch{
      setLabel('Failed to transcribe.');
    }
  }

  return (
    <View style={[backgrounStyle, styles.outer]}>
      <StatusBar barStyle={isDarkMode ? 'light-content': 'dark-content'}/>
      <ImageBackground source={DarkLogo} blurRadius={10} style={{height, width}}/>

      <Text style={styles.title}>{'Speech To Text'}</Text>

      {/* language */}
      <View style={styles.optionRow}>
        <Text style={styles.optionLabel}>Language:</Text> 
        {
          [['au', 'Auto'], ['en', 'EN'], ['fi', 'FI']].map(([val, display]) => (
            <TouchableOpacity key={val} onPress={()=>setLanguage(val)}
            style={[styles.chip, language === val && styles.chipActive]}
            >
              <Text style={styles.chipText}>{display}</Text>
            </TouchableOpacity>
          ))
        }
      </View>
      {/* model */}
      <View style={styles.modelRow}>
        <Text style={styles.optionLabel}>Model:</Text>
        {[['whisper','Whisper'],['wav2vec2','W2V'],['deepspeech2','DS2']].map(([val, display]) => (
          <TouchableOpacity key={val} onPress={() => setModelName(val)}
            style={[styles.chip, modelName === val && styles.chipActive]}>
            <Text style={styles.chipText}>{display}</Text>
          </TouchableOpacity>
        ))}
      </View>
      
      {/* Results */}
      <ScrollView style={styles.transcriptBox}>
        <Text style={styles.transcriptText}>
          {transcript || label || 'Press Record or pick an audio file'}
        </Text>
        {duration !== null && (
          <Text style={styles.durationText}>{duration.toFixed(1)}s</Text>
        )}
      </ScrollView>

      {/* Button */}
      <View style={styles.btnRow}>
        <TouchableOpacity onPress={ isRecording ? stopRecording: startRecording} 
          style={[styles.btn, isRecording && styles.btnStop]}
        >
          <Text style={styles.btnText}>{isRecording ? 'Stop': 'Record'}</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={pickAudioFile} style={styles.btn} disabled={isRecording}>
          <Text style={styles.btnText}>File</Text>
        </TouchableOpacity>
      </View>
    </View>
  )
}

export default App
const styles = StyleSheet.create({
  outer: { flex: 1, alignItems: 'center', justifyContent: 'center' },
  title: {
    position: 'absolute', top: Platform.OS === 'ios' ? 35 : 10,
    alignSelf: 'center', fontSize: 28, color: '#FFF', ...fonts.Bold,
  },
  optionRow: {
    flexDirection: 'row', alignItems: 'center',
    position: 'absolute', left: 20, right: 20,
    top: height * 0.15,   // ← thêm dòng này
  },
  modelRow: {
    flexDirection: 'row', alignItems: 'center',
    position: 'absolute', left: 20, right: 20,
    top: height * 0.23,   // ← thấp hơn language row
  },

  optionLabel: { color: '#FFF', ...fonts.Bold, marginRight: 8, fontSize: 14 },
  chip: {
    paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20,
    backgroundColor: 'rgba(255,255,255,0.3)', marginHorizontal: 4,
  },
  chipActive: { backgroundColor: '#FFF' },
  chipText: { fontSize: 13, ...fonts.Bold, color: '#000' },
  transcriptBox: {
    position: 'absolute', top: height * 0.35, left: 20, right: 20,
    maxHeight: height * 0.3, backgroundColor: 'rgba(0,0,0,0.5)',
    borderRadius: 12, padding: 12,
  },
  transcriptText: { color: '#FFF', fontSize: 16, ...fonts.Bold },
  durationText: { color: '#aaa', fontSize: 13, marginTop: 8 },
  btnRow: {
    position: 'absolute', bottom: 50,
    flexDirection: 'row', justifyContent: 'center', gap: 20,
  },
  btn: {
    backgroundColor: '#FFF', opacity: 0.9, paddingHorizontal: 30,
    paddingVertical: 16, borderRadius: 30,
  },
  btnStop: { backgroundColor: '#ff4444' },
  btnText: { fontSize: 18, ...fonts.Bold, color: '#000' },
});
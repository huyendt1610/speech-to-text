import { useState, useEffect, useRef } from "react"; 
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper,Tooltip, Switch, FormControlLabel, TextField, Box, Grid, TableContainer, 
  Table, TableBody, TableHead, TableRow, TableCell, CircularProgress, Select, MenuItem,InputLabel, FormControl } from "@material-ui/core";
import recordingLogo from "./src-images/music.png";
import stopRecordLogo from "./src-images/microphone.png";

import { DropzoneArea } from 'material-ui-dropzone';
import Clear from '@material-ui/icons/Clear';
import GetAppIcon from '@material-ui/icons/GetApp';

import useStyles from "./styles";
import ColorButton from "./ColorButton";
import { whisperLanguages } from "./language.js";


const axios = require("axios").default;
const controller = new AbortController();


export const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [audioFile, setAudioFile] = useState(false);
  const [isLoading, setIsloading] = useState(false);
  // const [checkedDeepSpeech2, setCheckedDeepSpeech2] = useState(false);

  //recording 
  const [recording, setRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const controllerRef = useRef(null);

  const [language, setLanguage] = useState("au");
  const [model, setModel] = useState("whisper");

  let wer = 0;

  const sendFile = async () => {
    controllerRef.current = new AbortController();
    try {
      if (audioFile) {
        //import.meta.env.
        //let url = checkedDeepSpeech2 ? process.env.REACT_APP_API_URL_DEEPSPEECH2 : process.env.REACT_APP_API_URL; 
        let formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("la", language);
        formData.append("model_name", model)
        let res = await axios({ // axe i os: make an http request to call backend 
          method: "post",
          url:process.env.REACT_APP_API_URL,
          data: formData,
          signal: controller.signal,
        });
        if (res.status === 200) {
          setData(res.data);
        }
        setIsloading(false);
      }
    } catch (err) {
      if (err.name === "CanceledError") {
        console.log("Upload canceled!");
      } else {
        console.error("Error:", err);
      }
    } finally {
      setIsloading(false);
    }
  }

  const cancelUpload = () => {
    if (controllerRef.current) controllerRef.current.abort();
    clearData(false)
  };

  const clearData = () => {
    setData(null);
    setAudioFile(false);
    setSelectedFile(null);
    setPreview(null);
    setIsloading(false)
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    setIsloading(true);
    sendFile();
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setAudioFile(false);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
    setAudioFile(true);
  };

  // if (data) {
  //   wer = (parseFloat(data.confidence) * 100).toFixed(2);
  // }

  // const handleChangeModel = (event) => {
  //   setCheckedDeepSpeech2(event.target.checked);
  // };

  const startRecording = async () => {
    clearData()
    setAudioFile(true);
    setIsloading(true);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    chunksRef.current = [];
    mediaRecorderRef.current.ondataavailable = e => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm" });
      //const url = URL.createObjectURL(blob);
      // setPreview(url);
      setData(undefined);
      setSelectedFile(blob);
      setAudioFile(true);
    };

    mediaRecorderRef.current.start();
    setRecording(true);
  };

  const stopRecording = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  const handleDownload = () => {
    if(data){
      // Tạo blob từ nội dung text
      const blob = new Blob([data.transcript], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);

      // Tạo thẻ <a> tạm và click
      const link = document.createElement('a');
      link.href = url;
      link.download = 'transcript.txt';
      link.click();

      // Giải phóng memory
      URL.revokeObjectURL(url);
    }
  };

  const handleSelectLanguage = (event) => {
      setLanguage(event.target.value)
  }

  const handleSelectModel = (event) => {
      setModel(event.target.value)
  }

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography className={classes.title} variant="h6" noWrap>
            STT: Speech-To-Text
          </Typography>
          <div className={classes.grow} />
          {/* <FormControlLabel
            control={
              <Switch
                checked={checkedDeepSpeech2}
                onChange={handleChangeModel} //change model 
              />
            }
            label="DeepSpeech2"
          /> */}
          Models
          <FormControl>
            <Select className={classes.selectWhite} value={model}
                    onChange={handleSelectModel} >
                <MenuItem value="whisper">Whisper</MenuItem>
                <MenuItem value="wav2vec2">Wav2Vec2</MenuItem>
                <MenuItem value="wav2vec2_fi">Wav2Vec2-Finnish</MenuItem>
                <MenuItem value="deepspeech2">DeepSpeech2</MenuItem>
            </Select>
          </FormControl>
          <FormControl>
            <Select className={classes.selectWhite} value={language}
                  onChange={handleSelectLanguage}>
              {Object.entries(whisperLanguages).map(([code, name]) => (
                <MenuItem key={code} value={code}>
                  {name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Tooltip title="Start recording">
            <Avatar src={ recording? recordingLogo: stopRecordLogo}         
              style={{cursor: "pointer" }} //record 
              onClick={recording ? stopRecording : startRecording}> 
            </Avatar>
          </Tooltip>
        </Toolbar>
      </AppBar>
      <Container maxWidth={false} className={classes.mainContainer} disableGutters={true}>
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!audioFile ? classes.imageCardEmpty : ''}`}>
              {audioFile && 
              // <CardActionArea>
              //   <CardMedia
              //     className={classes.media}
              //     image={preview}
              //     component="image"
              //     title="Contemplative Reptile"
              //   />
              // </CardActionArea>
              <audio controls src={preview} style={{ width: "100%" }} />
              }
              {!audioFile && <CardContent className={classes.content}>
                <DropzoneArea
                  // acceptedFiles={['image/*']}
                  acceptedFiles={['.mp3', '.wav', '.ogg', '.aac', '.flac']}
                  dropzoneText={"Drag and drop an audio file to process"}
                  onChange={onSelectFile}
                />
              </CardContent>}
              {data && <CardContent className={classes.detail}>
                <TableContainer component={Paper} className={classes.tableContainer}>
                  <Table className={classes.table} size="small" aria-label="simple table">
                    <TableHead className={classes.tableHead}>
                      <TableRow className={classes.tableRow}>
                        <TableCell className={classes.tableCell1}>Transcript:</TableCell>
                        {/* <TableCell align="right" className={classes.tableCell1}>Confidence:</TableCell> */}
                      </TableRow>
                    </TableHead>
                    <TableBody className={classes.tableBody}>
                      <TableRow className={classes.tableRow}>
                        <TableCell scope="row" className={classes.tableCell} style={{ height: 280 }}>
                          <Typography variant="h6">
                           <Box className={classes.transcriptBox}>
                              <Typography>
                                {data.transcript}
                              </Typography>
                            </Box>
                          </Typography>
                          <ColorButton variant="contained" color="primary" onClick={handleDownload} startIcon={<GetAppIcon />}>
                            Download
                          </ColorButton>           
                        </TableCell>
                      </TableRow>
                      <TableRow className={classes.tableRow}>
                        <TableCell scope="row" className={classes.tableCell1}>
                         Name: {data.filename.length > 38 ? "..." + data.filename.slice(-38) : data.filename}
                        </TableCell>
                      </TableRow>
                      <TableRow className={classes.tableRow}>
                        
                        <TableCell scope="row" className={classes.tableCell1}>
                           <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                            <span> Duration: {data.duration} sec</span>
                            <span> Language: {data.language}  </span>
                          </Box>
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>}
              {isLoading && 
              <CardContent className={classes.detail} style={{ height: 350 }}>
                <CircularProgress color="secondary" className={classes.loader} />
                <Typography className={classes.title} variant="h6" noWrap>
                  { recording ? "Recording" : "Processing"}
                </Typography>
              </CardContent>}
            </Card>
          </Grid>
          {(data || selectedFile)  &&
            <Grid item className={classes.buttonGrid} >
              <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="medium" onClick={clearData} startIcon={<Clear fontSize="large" />}>
                Clear
              </ColorButton>
            </Grid>
          }
          {/* {isLoading &&
            <Grid item className={classes.buttonGrid} >
              <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="large" onClick={cancelUpload} startIcon={<Clear fontSize="large" />}>
                Cancel
              </ColorButton>
            </Grid>
          } */}
          {recording &&
            <Grid item className={classes.buttonGrid} >
              <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="large" onClick={stopRecording} startIcon={<Clear fontSize="large" />}>
                Stop
              </ColorButton>
            </Grid>
          }
        </Grid >

      </Container >
    </React.Fragment >
  );
};

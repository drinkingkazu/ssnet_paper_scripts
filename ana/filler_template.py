
def get_template():
    return """
DataFiller: {

  Verbosity:    2
  EnableFilter: true
  RandomAccess: false
  UseThread:    false
  InputFiles: ["FILEPATH"]
  ProcessType:  ["SegFiller"]
  ProcessName:  ["SegFiller"]
    
  ProcessList: {
    SegFiller: {

      Verbosity: 2
      # DatumFillerBase configuration
      ImageProducer:  "wire"
      LabelProducer:  "segment"
      #WeightProducer: "spweight"
      # SimpleFiller configuration
      ClassTypeDef:      [0,0,0,3,3,3,6,6,6,6]
      ClassTypeList:     [3,6]
      Channels:          [PLANE]
      SegChannel:        2
      EnableMirror:      false
    }
  }
}
"""

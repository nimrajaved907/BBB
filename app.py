from io import StringIO
from Bio import SeqIO
import pandas as pd
import streamlit as st
from PIL import Image
import extractFeatures as fe
import numpy as np
import pickle
import base64

icon = Image.open('fav.png')
st.set_page_config(page_title='BBB', page_icon = icon)
with open("./scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("./model.pkl", 'rb') as file:
    model = pickle.load(file)

def seqValidator(seq):
    allowed_chars = set('ACDEFGHIKLMNPQRSTVWXY')
    if set(seq).issubset(allowed_chars):
        return True
    return False

final_df = pd.DataFrame(columns=['Sequence ID', 'Sequence', 'Label'])
seq = ""
st.header("""BBB""")

file_ = open("./WebPic.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()
st.markdown(f'<img src="data:image/gif;base64,{data_url}" alt="Header Image">', unsafe_allow_html=True)

st.sidebar.subheader(("Input Sequence(s) (FASTA FORMAT ONLY)"))
fasta_string  = st.sidebar.text_area("Sequence Input", height=200)
    
if st.button('Example'):
    st.code(">sp|O95476|CNEP1_HUMAN CTD nuclear envelope phosphatase 1 OS=Homo sapiens OX=9606 GN=CTDNEP1 PE=1 SV=2\nMMRTQCLLGLRTFVAFAAKLWSFFIYLLRRQIRTVIQYQTVRYDILPLSPVSRNRLAQVKRKILVLDLDETLIHSHHDGVLRPTVRPGTPPDFILKVVIDKHPVRFFVHKRPHVDFFLEVVSQWYELVVFTASMEIYGSAVADKLDNSRSILKRRYYRQHCTLELGSYIKDLSVVHSDLSSIVILDNSPGAYRSHPDNAIPIKSWFSDPSDTALLNLLPMLDALRFTADVRSVLSRNLHQHRLW", language="markdown")
    st.code(">sp|Q99653|CHP1_HUMAN Calcineurin B homologous protein 1 OS=Homo sapiens OX=9606 GN=CHP1 PE=1 SV=3\nMGSRASTLLRDEELEEIKKETGFSHSQITRLYSRFTSLDKGENGTLSREDFQRIPELAINPLGDRIINAFFPEGEDQVNFRGFMRTLAHFRPIEDNEKSKDVNGPEPLNSRSNKLHFAFRLYDLDKDEKISRDELLQVLRMMVGVNISDEQLGSIADRTIQEADQDGDSAISFTEFVKVLEKVDVEQKMSIRFLH", language="markdown")
    st.code(">sp|P32929|CGL_HUMAN Cystathionine gamma-lyase OS=Homo sapiens OX=9606 GN=CTH PE=1 SV=3\nMQEKDASSQGFLPHFQHFATQAIHVGQDPEQWTSRAVVPPISLSTTFKQGAPGQHSGFEYSRSGNPTRNCLEKAVAALDGAKYCLAFASGLAATVTITHLLKAGDQIICMDDVYGGTNRYFRQVASEFGLKISFVDCSKIKLLEAAITPETKLVWIETPTNPTQKVIDIEGCAHIVHKHGDIILVVDNTFMSPYFQRPLALGADISMYSATKYMNGHSDVVMGLVSVNCESLHNRLRFLQNSLGAVPSPIDCYLCNRGLKTLHVRMEKHFKNGMAVAQFLESNPWVEKVIYPGLPSHPQHELVKRQCTGCTGMVTFYIKGTLQHAEIFLKNLKLFTLAESLGGFESLAELPAIMTHASVLKNDRDVLGISDTLIRLSVGLEDEEDLLEDLDQALKAAHPPSGSHS", language="markdown")

    
if st.sidebar.button("SUBMIT"):
    if(fasta_string==""):
        st.info("Please input the sequence first.")
    fasta_io = StringIO(fasta_string) 
    records = SeqIO.parse(fasta_io, "fasta") 
    for rec in records:
        seq_id = str(rec.id)
        seq=str(rec.seq)
        if(seqValidator(seq)):
            df_temp = pd.DataFrame([[seq_id, seq,'None']], columns=['Sequence ID', 'Sequence','Label'] )
            final_df = pd.concat([final_df,df_temp], ignore_index=True)
        else:
            st.info("Sequence with Sequence ID: " + str(seq_id) + " is invalid, containing letters other than standard amino acids")
    fasta_io.close()
    if(final_df.shape[0]!=0):
        for iter in range(final_df.shape[0]):
            temp_seq =  final_df.iloc[iter, 1]
            fv_array = scaler.transform(np.array(fe.calcFV(temp_seq)).reshape(1, 153))
            score = model.predict(fv_array)
            pred_label = np.round_(score, decimals=0, out=None)
            if(pred_label==1):
                pred_label="BBB"
            else:
                pred_label="Non-BBB"
            final_df.iloc[iter, 2] = str(pred_label)

    st.dataframe(final_df)

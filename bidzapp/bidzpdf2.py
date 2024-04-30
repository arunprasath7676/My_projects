from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from openai import OpenAI
import os
import json

os.environ['OPENAI_API_KEY'] = "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"

example_prompt = PromptTemplate(
    input_variables=["input"], 
    template="Question: {input}\n"
)

# Here are your example inputs
inputs = {
    "input": """Chennai 
SUB: Quotation For Supply & Fixing of Civil works f or Office @ T.Nagar Date : 31.03.2021 
Site :  Chennai   Revision      00 
Kindly Thanks for your Valuable enquiry,we are send ing our competeive quotation as below,pls make a Or der as 
Description, 
S.NO. NATURE OF WORK QTY UNIT RATE AMOUNT BOQ Office :  No  1B, 3rd Street, 1st Main Road, Mutham il Nagar, Kallikuppam, Ambattur, Chennai   600 053 
Factory :  No   13, Small Street, Balaji Nagar, Pul iyambedu, Thiruverkadu, Chennai   600 077 
Quotation 
TO, 
Mr.Somasundharam 
Contact No: 
JOB Ref : ADI 03 2021 TRF 2020 2021 122 
DEAR SIR, 
the same, We hope that you will give a Chance to pr oof our Creativity ,Team work,Efficiency and Work F inishing 
Joinery Work 
1Demolition of Existing Flooring tiles with necessar y toolos and tackles etc, 
complete. Sq.ft 1035.00 38.00 39,330.00                         
2Demolition of Existing Toilet wall cladding and flo or tiles with necessary toolos 
and tackles etc, complete. Sq.ft 445.00 44.00 19,580.00                         
3 Deloition of Existing Kitchen Slaps and partition w all LS 1.00 6600.00 6,600.00                           
4 Applying and Laying of Water proofing in Toilet Are a Sq.ft 185.00 88.00 16,280.00                         
5 Pest control Anti termite treatment Sq.ft 1150.00 13.00 14,950.00                         
6 Removal of Debris from the Site Load 5.00 2640.00 13,200.00                         
7 Built in Ledge wall with plastering for WC Back wal l Sq.ft 35 250 8,750.00                           
8 Cement Plastering on wall in the toilet area Sq.ft 130 71 9,230.00                           
9 UPVC Ventilator For Toilet Nos 2 2690 5,380.00                           
10 Plumbing work with supply and laying of concealed i nlet Upvc and out lets PVC 
drain line with necessary wall chasing etic, comple te Toilet 2 20900 41,800.00                         
11 Plumbing Line to connect the external line Ls 1 2900 2,900.00                           
12 Dismantling & Removing the existing plumbing lines and seal the unwanted 
lines with necessary fittings Ls 1 2640 2,640.00                           
13 Supply and Laying of Vertified Tile Flooring of Siz e 600mm x 1200mm with 
necessary cement morter  considered M Sand  ratio o f 1: 6 and neceesry 
patterns with 3mm thick Spacer. The spacer groove s hall be grouted with joint 
fill adhesive including necessary manpower, lead an d lift etc, complete. Tile 
Basic cost Rs. 90  Sq.ft Sq.ft 1035.00 187.00 193,545.00                       
14 Supply and Laying of Anti Skid Tile Flooring of Siz e 600mm x 600mm with 
necessary cement morter  considered M Sand  ratio o f 1: 6 and neceesry 
patterns with 3mm thick Spacer. The spacer groove s hall be grouted with 
Epoxy joint fill adhesive including necessary manpo wer, lead and lift etc, 
complete. Tile Basic cost Rs. 60  Sq.ft Sq.ft 90.00 176.00 15,840.00                         
15 Supply and Laying of Wall Tile Cladding of Size 600 mm x 600mm with necessary 
cement morter  considered M Sand  ratio of 1: 6 and  neceesry patterns with 
3mm thick Spacer. The spacer groove shall be groute d with  joint fill adhesive 
including necessary manpower, lead and lift etc, co mplete. Tile Basic cost Rs. 
60  Sq.ft Sq.ft 330.00 170.00 56,100.00                          n16 Supply and Laying of Kitchen Wall Tile Cladding as per selection with necessary 
cement morter  considered M Sand  ratio of 1: 6 and  neceesry patterns with 
3mm thick Spacer. The spacer groove shall be groute d with  joint fill adhesive 
including necessary manpower, lead and lift etc, co mplete. Tile Basic cost Rs. 
55  Sq.ft Sq.ft 35.00 198.00 6,930.00                           
17 Toilet Fixtures 
18 Supply and fixing of cascade Model Eropian water cl oset from parryware   
Verve AM Nos 2 14850 29,700.00                         
19 Supply and fixing of Wall hung Wash Basin  from par ryware   ViVa Recta Nos 2 6160 12,320.00                         
20 Supply and fixing of healthfaucet   parryware   car diff with SS Hose Nos 2 187 374.00                              
21 Supply and fixing of Pillar cock for Wash basin   p arryware  Verve   T3902A1 Nos 2 2530 5,060.00                           
22 Supply and fixing of Two Way BiB cock for WC  parry ware  G3334A1 Nos 2 2530 5,060.00                           
23 Angle cock Nos 6 1078 6,468.00                           
24 Bottle Trap   Parryware T3203A1 Nos 3 1595 4,785.00                           
25 waste cupling Nos 3 495 1,485.00                           
26 Greatings Nos 4 385 1,540.00                           
27 connecting Braided hose Nos 3 440 1,320.00                           
28 Sink tap Nos 1 2035 2,035.00                           
29 SS SINK   Parryware C854399 flat edge   21  x 18  x  8  single Bowl Nos 1 6710 6,710.00                           
30 Extension Nipple Nos 8 195 1,560.00                           
31 POP Floor Protection Sq.ft 1035 11 11,385.00                         
32 Ceiling Paint with Plastic Emulsion Sq.ft 1100 18 19,800.00                         
33 Walls Paint with Premium Emulsion Sq.ft 2970 22 65,340.00                         
34 Enamal Painting on Windows and Grills LS 1 17600 17,600.00                         
645,597.00                   
25,000.00                     SPECIAL DISCOUNT TOTAL Amount 
620,597.00                   
1 50  Advance along with work order. 
2 40  After the material is delivered. 
3 10  After completion of work 
4 All the other Materials used will be of standard re puted make and sub. to sample Approvals & local mar ket availability. 
5
6 Power and water must be provided by the client with out cost. The manpower shall be accomodated at the site 
till the project duration time . Day and night acce ss to the site shall be provided to the manpower 
7
8 Hob, chimney, Oven & Sink, Electrical Work will be extra. 
9
THANKS & REGARDS, Quote considered without details drawing, drawing p roposal will be provided by AD inline with the quot e for the 
mutual understanding for both the parties 
Quotation is not including electrical, plumbing wor k, any other changes will be extra. 
V. Muthuvel Prabakaran 
00 91 995 270 5999 Grand Total Amount 
Any variation from the referred Scope of work will induce price revision and time. We are please to offer  Rs.6,20,597.50     Six Lakh s Twenty Thousand Five Hundred & Ninety Seven Only   Tax Extra 
TERMS & CONDITIONS :n,

prompt_data = extract the itemName, unitPrice, quantity, totalPrice, currency or similiar specifications and identify the common keys or attributes across the following text: Chennai 
SUB: Quotation For Supply & Fixing of Civil works f or Office @ T.Nagar Date : 31.03.2021 
Site :  Chennai   Revision      00 
Kindly Thanks for your Valuable enquiry,we are send ing our competeive quotation as below,pls make a Or der as 
Description, 
S.NO. NATURE OF WORK QTY UNIT RATE AMOUNT BOQ Office :  No  1B, 3rd Street, 1st Main Road, Mutham il Nagar, Kallikuppam, Ambattur, Chennai   600 053 
Factory :  No   13, Small Street, Balaji Nagar, Pul iyambedu, Thiruverkadu, Chennai   600 077 
Quotation 
TO, 
Mr.Somasundharam 
Contact No: 
JOB Ref : ADI 03 2021 TRF 2020 2021 122 
DEAR SIR, 
the same, We hope that you will give a Chance to pr oof our Creativity ,Team work,Efficiency and Work F inishing 
Joinery Work 
1Demolition of Existing Flooring tiles with necessar y toolos and tackles etc, 
complete. Sq.ft 1035.00 38.00 39,330.00                         
2Demolition of Existing Toilet wall cladding and flo or tiles with necessary toolos 
and tackles etc, complete. Sq.ft 445.00 44.00 19,580.00                         
3 Deloition of Existing Kitchen Slaps and partition w all LS 1.00 6600.00 6,600.00                           
4 Applying and Laying of Water proofing in Toilet Are a Sq.ft 185.00 88.00 16,280.00                         
5 Pest control Anti termite treatment Sq.ft 1150.00 13.00 14,950.00                         
6 Removal of Debris from the Site Load 5.00 2640.00 13,200.00                         
7 Built in Ledge wall with plastering for WC Back wal l Sq.ft 35 250 8,750.00                           
8 Cement Plastering on wall in the toilet area Sq.ft 130 71 9,230.00                           
9 UPVC Ventilator For Toilet Nos 2 2690 5,380.00                           
10 Plumbing work with supply and laying of concealed i nlet Upvc and out lets PVC 
drain line with necessary wall chasing etic, comple te Toilet 2 20900 41,800.00                         
11 Plumbing Line to connect the external line Ls 1 2900 2,900.00                           
12 Dismantling & Removing the existing plumbing lines and seal the unwanted 
lines with necessary fittings Ls 1 2640 2,640.00                           
13 Supply and Laying of Vertified Tile Flooring of Siz e 600mm x 1200mm with 
necessary cement morter  considered M Sand  ratio o f 1: 6 and neceesry 
patterns with 3mm thick Spacer. The spacer groove s hall be grouted with joint 
fill adhesive including necessary manpower, lead an d lift etc, complete. Tile 
Basic cost Rs. 90  Sq.ft Sq.ft 1035.00 187.00 193,545.00                       
14 Supply and Laying of Anti Skid Tile Flooring of Siz e 600mm x 600mm with 
necessary cement morter  considered M Sand  ratio o f 1: 6 and neceesry 
patterns with 3mm thick Spacer. The spacer groove s hall be grouted with 
Epoxy joint fill adhesive including necessary manpo wer, lead and lift etc, 
complete. Tile Basic cost Rs. 60  Sq.ft Sq.ft 90.00 176.00 15,840.00                         
15 Supply and Laying of Wall Tile Cladding of Size 600 mm x 600mm with necessary 
cement morter  considered M Sand  ratio of 1: 6 and  neceesry patterns with 
3mm thick Spacer. The spacer groove shall be groute d with  joint fill adhesive 
including necessary manpower, lead and lift etc, co mplete. Tile Basic cost Rs. 
60  Sq.ft Sq.ft 330.00 170.00 56,100.00                          n16 Supply and Laying of Kitchen Wall Tile Cladding as per selection with necessary 
cement morter  considered M Sand  ratio of 1: 6 and  neceesry patterns with 
3mm thick Spacer. The spacer groove shall be groute d with  joint fill adhesive 
including necessary manpower, lead and lift etc, co mplete. Tile Basic cost Rs. 
55  Sq.ft Sq.ft 35.00 198.00 6,930.00                           
17 Toilet Fixtures 
18 Supply and fixing of cascade Model Eropian water cl oset from parryware   
Verve AM Nos 2 14850 29,700.00                         
19 Supply and fixing of Wall hung Wash Basin  from par ryware   ViVa Recta Nos 2 6160 12,320.00                         
20 Supply and fixing of healthfaucet   parryware   car diff with SS Hose Nos 2 187 374.00                              
21 Supply and fixing of Pillar cock for Wash basin   p arryware  Verve   T3902A1 Nos 2 2530 5,060.00                           
22 Supply and fixing of Two Way BiB cock for WC  parry ware  G3334A1 Nos 2 2530 5,060.00                           
23 Angle cock Nos 6 1078 6,468.00                           
24 Bottle Trap   Parryware T3203A1 Nos 3 1595 4,785.00                           
25 waste cupling Nos 3 495 1,485.00                           
26 Greatings Nos 4 385 1,540.00                           
27 connecting Braided hose Nos 3 440 1,320.00                           
28 Sink tap Nos 1 2035 2,035.00                           
29 SS SINK   Parryware C854399 flat edge   21  x 18  x  8  single Bowl Nos 1 6710 6,710.00                           
30 Extension Nipple Nos 8 195 1,560.00                           
31 POP Floor Protection Sq.ft 1035 11 11,385.00                         
32 Ceiling Paint with Plastic Emulsion Sq.ft 1100 18 19,800.00                         
33 Walls Paint with Premium Emulsion Sq.ft 2970 22 65,340.00                         
34 Enamal Painting on Windows and Grills LS 1 17600 17,600.00                         
645,597.00                   
25,000.00                     SPECIAL DISCOUNT TOTAL Amount 
620,597.00                   
1 50  Advance along with work order. 
2 40  After the material is delivered. 
3 10  After completion of work 
4 All the other Materials used will be of standard re puted make and sub. to sample Approvals & local mar ket availability. 
5
6 Power and water must be provided by the client with out cost. The manpower shall be accomodated at the site 
till the project duration time . Day and night acce ss to the site shall be provided to the manpower 
7
8 Hob, chimney, Oven & Sink, Electrical Work will be extra. 
9
THANKS & REGARDS, Quote considered without details drawing, drawing p roposal will be provided by AD inline with the quot e for the 
mutual understanding for both the parties 
Quotation is not including electrical, plumbing wor k, any other changes will be extra. 
V. Muthuvel Prabakaran 
00 91 995 270 5999 Grand Total Amount 
Any variation from the referred Scope of work will induce price revision and time. We are please to offer  Rs.6,20,597.50     Six Lakh s Twenty Thousand Five Hundred & Ninety Seven Only   Tax Extra 
TERMS & CONDITIONS :  
 n and represent them in a only provide a rfc8259 compliant json response without deviation the following key: itemName, unitPrice, quantity, totalPrice, currency and other specifications also. identify and extract item description or sentence that starting with either numerals (1, 2, 3, etc.) or alphabet letters (a, b, c, etc.) or have other serial number criteria. recognize each item description or sentence as separate keys or attributes from Chennai 
SUB: Quotation For Supply & Fixing of Civil works f or Office @ T.Nagar Date : 31.03.2021 
Site :  Chennai   Revision      00 
Kindly Thanks for your Valuable enquiry,we are send ing our competeive quotation as below,pls make a Or der as 
Description, 
S.NO. NATURE OF WORK QTY UNIT RATE AMOUNT BOQ Office :  No  1B, 3rd Street, 1st Main Road, Mutham il Nagar, Kallikuppam, Ambattur, Chennai   600 053 
Factory :  No   13, Small Street, Balaji Nagar, Pul iyambedu, Thiruverkadu, Chennai   600 077 
Quotation 
TO, 
Mr.Somasundharam 
Contact No: 
JOB Ref : ADI 03 2021 TRF 2020 2021 122 
DEAR SIR, 
the same, We hope that you will give a Chance to pr oof our Creativity ,Team work,Efficiency and Work F inishing 
Joinery Work 
1Demolition of Existing Flooring tiles with necessar y toolos and tackles etc, 
complete. Sq.ft 1035.00 38.00 39,330.00                         
2Demolition of Existing Toilet wall cladding and flo or tiles with necessary toolos 
and tackles etc, complete. Sq.ft 445.00 44.00 19,580.00                         
3 Deloition of Existing Kitchen Slaps and partition w all LS 1.00 6600.00 6,600.00                           
4 Applying and Laying of Water proofing in Toilet Are a Sq.ft 185.00 88.00 16,280.00                         
5 Pest control Anti termite treatment Sq.ft 1150.00 13.00 14,950.00                         
6 Removal of Debris from the Site Load 5.00 2640.00 13,200.00                         
7 Built in Ledge wall with plastering for WC Back wal l Sq.ft 35 250 8,750.00                           
8 Cement Plastering on wall in the toilet area Sq.ft 130 71 9,230.00                           
9 UPVC Ventilator For Toilet Nos 2 2690 5,380.00                           
10 Plumbing work with supply and laying of concealed i nlet Upvc and out lets PVC 
drain line with necessary wall chasing etic, comple te Toilet 2 20900 41,800.00                         
11 Plumbing Line to connect the external line Ls 1 2900 2,900.00                           
12 Dismantling & Removing the existing plumbing lines and seal the unwanted 
lines with necessary fittings Ls 1 2640 2,640.00                           
13 Supply and Laying of Vertified Tile Flooring of Siz e 600mm x 1200mm with 
necessary cement morter  considered M Sand  ratio o f 1: 6 and neceesry 
patterns with 3mm thick Spacer. The spacer groove s hall be grouted with joint 
fill adhesive including necessary manpower, lead an d lift etc, complete. Tile 
Basic cost Rs. 90  Sq.ft Sq.ft 1035.00 187.00 193,545.00                       
14 Supply and Laying of Anti Skid Tile Flooring of Siz e 600mm x 600mm with 
necessary cement morter  considered M Sand  ratio o f 1: 6 and neceesry 
patterns with 3mm thick Spacer. The spacer groove s hall be grouted with 
Epoxy joint fill adhesive including necessary manpo wer, lead and lift etc, 
complete. Tile Basic cost Rs. 60  Sq.ft Sq.ft 90.00 176.00 15,840.00                         
15 Supply and Laying of Wall Tile Cladding of Size 600 mm x 600mm with necessary 
cement morter  considered M Sand  ratio of 1: 6 and  neceesry patterns with 
3mm thick Spacer. The spacer groove shall be groute d with  joint fill adhesive 
including necessary manpower, lead and lift etc, co mplete. Tile Basic cost Rs. 
60  Sq.ft Sq.ft 330.00 170.00 56,100.00                          n16 Supply and Laying of Kitchen Wall Tile Cladding as per selection with necessary 
cement morter  considered M Sand  ratio of 1: 6 and  neceesry patterns with 
3mm thick Spacer. The spacer groove shall be groute d with  joint fill adhesive 
including necessary manpower, lead and lift etc, co mplete. Tile Basic cost Rs. 
55  Sq.ft Sq.ft 35.00 198.00 6,930.00                           
17 Toilet Fixtures 
18 Supply and fixing of cascade Model Eropian water cl oset from parryware   
Verve AM Nos 2 14850 29,700.00                         
19 Supply and fixing of Wall hung Wash Basin  from par ryware   ViVa Recta Nos 2 6160 12,320.00                         
20 Supply and fixing of healthfaucet   parryware   car diff with SS Hose Nos 2 187 374.00                              
21 Supply and fixing of Pillar cock for Wash basin   p arryware  Verve   T3902A1 Nos 2 2530 5,060.00                           
22 Supply and fixing of Two Way BiB cock for WC  parry ware  G3334A1 Nos 2 2530 5,060.00                           
23 Angle cock Nos 6 1078 6,468.00                           
24 Bottle Trap   Parryware T3203A1 Nos 3 1595 4,785.00                           
25 waste cupling Nos 3 495 1,485.00                           
26 Greatings Nos 4 385 1,540.00                           
27 connecting Braided hose Nos 3 440 1,320.00                           
28 Sink tap Nos 1 2035 2,035.00                           
29 SS SINK   Parryware C854399 flat edge   21  x 18  x  8  single Bowl Nos 1 6710 6,710.00                           
30 Extension Nipple Nos 8 195 1,560.00                           
31 POP Floor Protection Sq.ft 1035 11 11,385.00                         
32 Ceiling Paint with Plastic Emulsion Sq.ft 1100 18 19,800.00                         
33 Walls Paint with Premium Emulsion Sq.ft 2970 22 65,340.00                         
34 Enamal Painting on Windows and Grills LS 1 17600 17,600.00                         
645,597.00                   
25,000.00                     SPECIAL DISCOUNT TOTAL Amount 
620,597.00                   
1 50  Advance along with work order. 
2 40  After the material is delivered. 
3 10  After completion of work 
4 All the other Materials used will be of standard re puted make and sub. to sample Approvals & local mar ket availability. 
5
6 Power and water must be provided by the client with out cost. The manpower shall be accomodated at the site 
till the project duration time . Day and night acce ss to the site shall be provided to the manpower 
7
8 Hob, chimney, Oven & Sink, Electrical Work will be extra. 
9
THANKS & REGARDS, Quote considered without details drawing, drawing p roposal will be provided by AD inline with the quot e for the 
mutual understanding for both the parties 
Quotation is not including electrical, plumbing wor k, any other changes will be extra. 
V. Muthuvel Prabakaran 
00 91 995 270 5999 Grand Total Amount 
Any variation from the referred Scope of work will induce price revision and time. We are please to offer  Rs.6,20,597.50     Six Lakh s Twenty Thousand Five Hundred & Ninety Seven Only   Tax Extra 
TERMS & CONDITIONS :  
 n. each qualifying item description or sentence should be captured entirely and presented against the respective somewhat matching with itemName, unitPrice, quantity, totalPrice, currency or similar specifications. provide a comprehensive, side-by-side, and horizontal comparison of these in json format. verify the accuracy of the data before finalizing the comparison chart. repeat the process in case it fails to extract the common keys or specifications or attributes in the first attempt. """
}

# Create chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", example_prompt.template.format(**inputs)),
    ("human", inputs["input"]),
])

# Format the messages
formatted_messages = chat_prompt.format_messages()

if len(formatted_messages) >= 2:
    generated_prompt_text = formatted_messages[1].content

    # Call OpenAI API for chat completion
    client = OpenAI()
    response = client.chat.completions.create(
        model="text-davinci-003",
        messages=[
            {"role": "system", "content": generated_prompt_text}
        ]
    )
    print(response)
else:
    print("Error: Not enough messages in formatted_messages to access index 1.")
## About
This is a project created by Cooper Bowden in collaboration with Christopher Newport University for the city of Newport 
News to create a AI chatbot to explain city code and other city resources using open source projects. 
The project primarily uses vllm, llama-index, and streamlit; all of which are open-source libraries maintained by the community.

## Initial Setup
- download libraries from requirements.txt file
- move streamlit config file to .streamlit 
- replace write.py at .venv/Lib/site-packages/streamlit/elements/write.py and add "awrite_stream = _main.awrite_stream" to .venv/Lib/site-packages/streamlit/__init__.py
- start vllm server with "vllm serve microsoft/phi-4 --config config/vllm/config.yaml"
- start streamlit server with "streamlit run web_ui.py"
- open server with "localhost:8501"

## Customization
- New information can be added to the chatbot through the excel file in data/city_code/city_code_of_ordinances using the same format
- The chatbot model can be changed by specifying a different model with vllm serve {model name}. Model must be from huggingface repository. 
- The system prompt can be changed in web_ui.py
- The temperature can be changed in the OpenAILike in run.py. Values closer to 1 make the responses more creative and values closer to 0 make  the responses more predictable


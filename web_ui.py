import streamlit as st
from llm import ChatEng
import asyncio
from asyncio import run_coroutine_threadsafe
from threading import Thread
import nest_asyncio
import random
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from typing import Optional
from streamlit_feedback import streamlit_feedback


class GUI:
    def __init__(self):
        st.set_page_config(page_title="NN AI", page_icon="assets/nn_icon.jpg", menu_items={
            'Report a bug': "mailto:cooper.bowden.23@cnu.edu",
            'About': "# Newport News AI Chatbot.\n  "
                     "Learn about the City Code and other resources the city offers!"
        })
        self.model = self.load_model()
        st.session_state.event_loop, worker_thread = self.create_loop()
        self.system_prompt = ("I am a citizen of Newport News, Virginia asking about the city code. "
                              "You are a chatbot for the city of Newport News that answers the user's question. "
                              "Using only the provided context, answer the user's query. If the answer cannot be "
                              "found within the given context, state that you do not have enough information. "
                              "Do not say 'According to' or anything similar. "
                              "Do not include links in your response unless directly provided by the system prompt. "
                              "additional resources:" + self.model.read_additional_resources()
                              )
        self.all_questions = self.read_all_questions()
        # Only runs for first time startup
        if "suggested_questions" not in st.session_state:
            st.session_state.suggested_questions = random.sample(self.all_questions, 5)
        if "feedback_key" not in st.session_state:
            st.session_state.feedback_key = 0
        self.create_gui()
        self.display_messages()

    def create_gui(self):
        """initial GUI set up"""
        # create page
        st.logo("assets/alt_logo.png", size="large", icon_image="assets/nn_icon.jpg")
        # load model
        if len(st.session_state) == 3:
            st.session_state["messages"] = []
            st.session_state["assistant"] = self.model.create_engine(self.system_prompt)

        st.session_state["ingestion_spinner"] = st.empty()
        with st.sidebar:
            self.reset_button = st.button(label="New Chat", type='primary', use_container_width=True)
            self.get_random = st.button(label="Generate New Questions", type="primary", use_container_width=True)
            self.stream_mode = True
        if self.reset_button:
            self.reset_conversation()
        if self.get_random:
            st.session_state.suggested_questions = st.session_state.suggested_questions = (
                random.sample(self.all_questions, 5))
        # Edit sidebar html
        self.sidebar_html()
        # Create footer
        self.footer()

    def display_messages(self):
        """create chat message display and update messages"""
        st.subheader("Newport News AI Chatbot", anchor=False)
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                with st.chat_message("assistant", avatar="assets/nn_icon.jpg"):
                    st.markdown(message["content"])
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
        # Change color of text in chat box
        self.chat_input_html()

        pills_placeholder = st.empty()
        query = pills_placeholder.pills(label="suggested questions", options=st.session_state.suggested_questions,
                                        label_visibility="collapsed")
        if prompt := st.chat_input("Enter a Question"):
            # Hide pills while generating response
            pills_placeholder.empty()
            self.generate_response(prompt)
        elif query:
            # Hide pills while generating response
            pills_placeholder.empty()
            self.generate_response(query)

        if len(st.session_state.messages) != 0:
            # Only display if messages are displayed
            feedback = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[optional] Please provide feedback",
                key=f"feedback_{st.session_state.feedback_key}",
            )
        else:
            feedback = None
        if feedback:
            self.send_feedback(feedback)

    def generate_questions(self, nodes, max_questions, query=None, question_prompt: Optional[str] = None,):
        """ generate suggested questions for each node, use web_ui function rather than llm function to use same
        event loop
        :param nodes:
        :type nodes: TextNode List
        :param max_questions: max amount of questions to be generated
        :type max_questions: int
        :param query: user query
        :type query: string
        :param question_prompt: prompt to generate questions
        :return: List of suggested questions
        """
        nodes = nodes[0:max_questions]
        question_prompt = (question_prompt or
                           ("You are an assistant. Your task is to setup 1 question "
                            "that the user might ask. Assume the user has no knowledge of anything in the context. "
                            "The questions should be diverse in nature across the "
                            "document. Restrict the questions to the context information provided."
                            "Do not reference the section or chapter. Keep the response brief and concise. "
                            "Make the questions similar to the user query, but not the same as the user query"
                            f"user query: {query}"))
        dataset_generator = RagDatasetGenerator.from_documents(
            documents=nodes,
            llm=self.model.llm,
            num_questions_per_chunk=1,
            question_gen_query=question_prompt,
        )
        questions = run_coroutine_threadsafe(dataset_generator.agenerate_questions_from_nodes(),
                                             st.session_state.event_loop).result()
        return questions

    def generate_response(self, prompt):
        no_response_message = ("I'm sorry, I could not find any relevant information to your query. "
                               "please ask another question!")
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        loader = st.empty()
        try:
            if self.stream_mode:
                with st.spinner("Generating Response..."):
                    stream_response = run_coroutine_threadsafe(st.session_state["assistant"].astream_chat(prompt),
                                                               st.session_state.event_loop).result()
                with st.chat_message("assistant", avatar="assets/nn_icon.jpg"):
                    chat_response = asyncio.run(st.awrite_stream(stream_response))
                    if chat_response == "Empty Response":
                        # Write custom message if no nodes found and empty response received
                        chat_response = no_response_message
                        st.markdown(chat_response)
                source_nodes = stream_response.source_nodes
            else:
                with loader.status("Generating Response... "):
                    chat_response = run_coroutine_threadsafe(st.session_state["assistant"].achat(prompt),
                                                             st.session_state.event_loop).result()
                    chat_response.response = chat_response.response.replace("$", r"\$")
                    st.markdown(chat_response)
                    source_nodes = chat_response.source_nodes
                    if chat_response.response == "Empty Response":
                        chat_response = no_response_message
                with st.chat_message("assistant"):
                    st.markdown(chat_response)

            loader.empty()
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": chat_response})
            # Display assistant response in chat message container
            self.display_references(self.model.get_metadata(source_nodes))
            nodes = []
            for source_node in source_nodes:
                nodes.append(source_node.node)

            with st.spinner("Generating Questions..."):
                st.session_state.suggested_questions = self.generate_questions(nodes, max_questions=3, query=prompt)
            # Create new pills section below response
            st.pills(label="suggested questions",
                     options=st.session_state.suggested_questions,
                     label_visibility="collapsed")
        except Exception as e:
            # Print exception to terminal w/o sending exception to UI
            print(e)
            st.error("Unable to generate response. Please try again later.")

    def display_references(self, metadata):
        """display metadata on sidebar
        :param metadata: URL, section, title, and score of returned nodes
        :type metadata: dict
        """
        url = []
        title = []
        subtitle = []
        relevance_score = []
        for data in metadata:
            url.append(str(data["url"]))
            title.append(str(data["Title"]))
            subtitle.append(str(data["Subtitle"]))
            relevance_score.append(str(data["Score"]))
        with st.sidebar:
            # Changes color of text to white and centers it
            st.html("""
                <h1 style="color:white"> References</h1>
                <style>
                h1 {text-align: center;}
                </style>
            """)
            for i in range(len(url)):
                relevance_percent = str(int(float((relevance_score[i])) * 100))
                if subtitle[i] == "nan":
                    title_string = title[i]
                else:
                    title_string = title[i] + ": " + subtitle[i]

                if url[i] == "nan":
                    string_to_display = f"""
                        <span style="color:white">{title_string}</span>
                        <span style="color:white">- Score {relevance_percent}%</span>
                        """
                else:
                    string_to_display = f"""
                        <a style="color:white" href="{url[i]}", target="_blank")>{title_string}</a>
                        <span style="color:white">- Score {relevance_percent}%</span>
                        """
                st.html(string_to_display)

    def reset_conversation(self):
        """run chat engine reset and reset streamlit message history"""
        st.session_state["assistant"].reset()
        st.session_state.messages = []
        st.session_state.query = None
        st.session_state.suggested_questions = st.session_state.suggested_questions = random.sample(self.all_questions,
                                                                                                    5)

    def send_feedback(self,feedback):
        """ send feedback through API, currently not connected to anything
        :param feedback: rating, user entered feedback, message history, and most recent questions
        :type feedback: dictionary
        """
        print("SEND FEEDBACK")
        st.toast("Thanks, Your feedback has been received! NOTE: currently does not send feedback anywhere nor save it",
                 icon=":material/check_circle:")
        # Add user messages to feedback
        feedback["messages"] = st.session_state.messages
        feedback["questions"] = st.session_state.suggested_questions
        # SENDING FEEDBACK NEEDS TO BE SET UP
        # set new feedback key
        st.session_state.feedback_key += 1

    @st.cache_resource
    def read_all_questions(_self):
        """read questions from text file
        :return: suggested questions
        :rtype: list
        """
        questions = _self.model.read_questions()
        return questions

    @st.cache_resource
    def load_model(_self):
        """create and return chat model
        :return model: LLM chat engine that can be queried
        :rtype model: ChatEng
        """
        print("created model")
        model = ChatEng()
        return model

    @st.cache_resource(show_spinner=False)
    def create_loop(_self):
        """create event loop and cache it, allows for single event loop to be used by vllm"""
        nest_asyncio.apply()
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever)
        thread.start()
        return loop, thread

    def chat_input_html(self):
        """ Edit html of chat input box"""
        st.html("""
            <style>
            [data-testid="stChatInputTextArea"] {
            color: #FFF;
            caret-color: #FFF;
            font-size: 15px;

            }
            [data-testid="stChatInputTextArea"]::placeholder {
              color: white;
              font-size: 15px;
              bottom: 50;
            }
            [data-testid="stBottom"] {
            height: 50px;
            bottom: 110px;

            }
            [data-testid="stBottomBlockContainer"] {
            height: 50px;
            bottom: 110px;
            }
            div[data-testid="stMainBlockContainer"] {
            top: 200px;
            margin-bottom: 90px;
            }
            </style>
            """)

    def sidebar_html(self):
        st.html("""<style>
          div[data-testid="stSidebarHeader"] > img, div[data-testid="collapsedControl"] > img {
              height: 5rem;
              width: auto;
              text-align: center;
              display: block;
              margin-left: auto;
              margin-right: auto;
              width: 100%;
          }
          """)

    def footer(self):
        """Create footer under chat input box using html"""
        ft = """
        <style>
        a:link , a:visited{
        color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
        background-color: transparent;
        text-decoration: underline;
        }

        a:hover,  a:active {
        color: #de5647; /* theme's primary color*/
        background-color: transparent;
        text-decoration: underline;
        }

        #page-container {
          position: relative;
          min-height: 10vh;
        }

        footer{
            visibility:hidden;
        }

        .footer {
        margin-left: -20px;
        max-width: 736px;
        z-index: 1000;
        position: fixed;
        width: 100%;
        bottom: 0;
        max-height: 110px;
        padding-bottom: 16px;
        padding-left: 16px;
        padding-right: 16px;
        padding-top: 16px;
        background-color: white;
        color: #808080; /* theme's text color hex code at 50 percent brightness*/
        text-align: center; /* you can replace 'left' with 'center' or 'right' if you want*/
          }

        </style>

        <div id="page-container">
        <div class="footer">
        <p align="center" style='font-size: 0.9em;' >Responses may occasionally produce inaccurate or incomplete 
        content, Verify results with references in the sidebar</p>
        <span align="center" style='font-size: 1em;' >Please report any issues</span>
        <a color="blue" align="center" style='font-size: 1em;' href="https://www.nnva.gov/2178/Contact"> HERE</a>
        </div>
        </div>


        """
        st.html(ft)


if __name__ == "__main__":
    gui = GUI()

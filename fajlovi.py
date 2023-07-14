def read_pdf(file):
    pdf_reader = pypdf.PdfReader(file)
    num_pages = len(pdf_reader.pages)
    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

#file loader
st.title("Sova demo app")
uploaded_files = st.file_uploader("Choose file(s)", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        filename = './docs_database/' + uploaded_file.name
        if '.pdf' in filename:
            pdf_converted_to_text = read_pdf(uploaded_file)
            pdf_converted_to_text = pdf_converted_to_text.encode("ascii", "replace")
            pdf_converted_to_text = pdf_converted_to_text.decode(encoding="utf-8", errors="ignore")
            pdf_converted_to_text = pdf_converted_to_text.replace("?", " ")
            filename = filename.replace('.pdf', '.txt')
            with open(filename, 'wt', encoding="utf-8") as f:
                f.write(pdf_converted_to_text)
        else:
            file = uploaded_file.getvalue()
            with open(filename, 'wb') as f:
                f.write(file)
        st.write("File "+uploaded_file.name+" loaded")

#odabir modela 
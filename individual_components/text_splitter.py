text='''Interstellar is a 2014 epic science fiction film directed by Christopher Nolan, who co-wrote the screenplay with his brother Jonathan. It features an ensemble cast led by Matthew McConaughey, Anne Hathaway, Jessica Chastain, Bill Irwin, Ellen Burstyn, and Michael Caine. Set in a dystopian future where Earth is suffering from catastrophic blight and famine, the film follows a group of astronauts who travel through a wormhole near Saturn in search of a new home for mankind.

The screenplay had its origins in a script that Jonathan had developed in 2007 and was originally set to be directed by Steven Spielberg. Theoretical physicist Kip Thorne was an executive producer and scientific consultant on the film, and wrote the tie-in book The Science of Interstellar. It was Lynda Obst's final film as producer before her death. Cinematographer Hoyte van Hoytema shot it on 35 mm film in the Panavision anamorphic format and IMAX 70 mm. Filming began in late 2013 and took place in Alberta, Klaustur, and Los Angeles. Interstellar uses extensive practical and miniature effects, and the company DNEG created additional digital effects.

Interstellar premiered at the TCL Chinese Theatre on October 26, 2014, and was released in theaters in the United States on November 5, and in the United Kingdom on November 7. In the United States, it was first released on film stock, expanding to venues using digital projectors. The film received generally positive reviews from critics and was a commercial success, grossing $681 million worldwide during its initial theatrical run, and $758.6 million worldwide with subsequent releases, making it the tenth-highest-grossing film of 2014. Among its various accolades, Interstellar was nominated for five awards at the 87th Academy Awards, winning Best Visual Effects.'''

from langchain.text_splitter import RecursiveCharacterTextSplitter

r_splitter= RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " "],
    chunk_size=200,
    chunk_overlap=100,
)

chunks=r_splitter.split_text(text)
print(f"LENGTH OF CHUNKS IS: {len(chunks)}")

print("*********************")

for chunk in chunks:

    print(len(chunk))
    # print("\n\n")

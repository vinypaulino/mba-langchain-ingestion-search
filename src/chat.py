from search import search_prompt


def main():
    try:
        chain = search_prompt()
    except Exception as error:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        print(error)
        return

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat pronto. Digite sua pergunta ou 'sair' para encerrar.")
    while True:
        question = input("\nPERGUNTA: ").strip()
        if question.lower() in ("sair", "exit", "quit"):
            break
        if not question:
            continue
        print(f"RESPOSTA: {chain(question)}")


if __name__ == "__main__":
    main()

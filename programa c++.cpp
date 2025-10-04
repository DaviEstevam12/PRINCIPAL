#include <iostream>
#include <string>

std::string classificarPessoa(int idade, std::string pais) {
    std::string faixaEtaria;
    if (idade < 12) {
        faixaEtaria = "Criança";
    } else if (idade <= 17) {
        faixaEtaria = "Adolescente";
    } else {
        faixaEtaria = "Adulta";
    }

    std::string nacionalidade;
    if (pais == "Brasil") {
        nacionalidade = "Brasileira";
    } else {
        nacionalidade = "Estrangeira";
    }

    return faixaEtaria + " " + nacionalidade;
}

int main() {
    std::cout << classificarPessoa(17, "Argentina") << std::endl; // Adolescente Estrangeira
    std::cout << classificarPessoa(10, "Brasil") << std::endl;    // Criança Brasileira
    std::cout << classificarPessoa(25, "Chile") << std::endl;     // Adulta Estrangeira

    return 0;
}


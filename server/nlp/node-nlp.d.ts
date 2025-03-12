declare module 'node-nlp' {
  export class NlpManager {
    constructor(options: { languages: string[] });
    process(language: string, text: string): Promise<any>;
  }
}
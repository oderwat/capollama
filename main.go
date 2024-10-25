package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/alexflint/go-arg"
	"github.com/ollama/ollama/api"
)

func GenerateWithImage(ol *api.Client, model, prompt, imagePath string) (string, error) {
	// First, convert the image to base64
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image: %w", err)
	}

	msg := api.Message{
		Role:    "user",
		Content: prompt,
		Images:  []api.ImageData{imageData},
	}

	ctx := context.Background()
	req := &api.ChatRequest{
		Model:    model,
		Messages: []api.Message{msg},
	}

	var response strings.Builder
	respFunc := func(resp api.ChatResponse) error {
		response.WriteString(resp.Message.Content)
		return nil
	}

	err = ol.Chat(ctx, req, respFunc)
	if err != nil {
		log.Fatal(err)
	}
	return response.String(), nil
}

// ProcessImages walks through a given path and processes image files
func ProcessImages(path string, processFunc func(imagePath, rootDir string)) error {
	// Get file info
	fileInfo, err := os.Stat(path)
	if err != nil {
		return err // Silently ignore errors
	}

	// If it's a single file, process it if it's an image
	if !fileInfo.IsDir() {
		if isImageFile(path) {
			// For single files, use the parent directory as root
			rootDir := filepath.Dir(path)
			processFunc(path, rootDir)
		}
		return nil
	}

	// For directories, walk through all files recursively
	rootDir := path // Store the top-level directory
	err = filepath.Walk(path, func(currentPath string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // Continue walking despite errors
		}

		// Skip hidden directories (starting with .)
		if info.IsDir() {
			base := filepath.Base(currentPath)
			if strings.HasPrefix(base, ".") {
				return filepath.SkipDir
			}
		}

		if !info.IsDir() && isImageFile(currentPath) {
			processFunc(currentPath, rootDir)
		}
		return nil
	})
	return err
}

// isImageFile checks if the file has an image extension
func isImageFile(path string) bool {
	ext := strings.ToLower(filepath.Ext(path))
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png"
}

type Args struct {
	Path          string `arg:"positional,required" help:"Path to an image or a directory with images"`
	WriteCaptions bool   `arg:"--write-caption,-w" help:"Write captions as .txt (stripping the original extension)"`
	StartCaption  string `arg:"--start,-s" help:"Start the caption with this (image of Leela the dog,)"`
	EnddCaption   string `arg:"--end,-e" help:"End the caption with this (in the style of 'something')"`
	Prompt        string `arg:"--prompt,-p" help:"The prompt to use" default:"Please describe the content and style of this image in detail. Answer only with one sentence that is starting with \"A ...\""`
	Model         string `arg:"--model,-m" help:"The model that will be used (must be a vision model like \"llava\")" default:"x/llama3.2-vision"`
}

func main() {
	var args Args

	arg.MustParse(&args)

	ol, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	//  and mention "colorized photo"
	err = ProcessImages(args.Path, func(path string, root string) {
		captionText, err := GenerateWithImage(ol, args.Model, args.Prompt, path)
		if err != nil {
			log.Fatalf("Aborting because of %v", err)
		}
		captionText = args.StartCaption + captionText + args.EnddCaption
		fmt.Printf("%s: %s\n", strings.TrimPrefix(path, root), captionText)
		captionFile := strings.TrimSuffix(path, filepath.Ext(path)) + ".txt"
		if args.WriteCaptions {
			err = os.WriteFile(captionFile, []byte(captionText), 0644)
			if err != nil {
				log.Fatalf("Could not write file %q", err)
			}
		}
	})
	if err != nil {
		log.Printf("Error: %s", err.Error())
		os.Exit(1)
	}
}

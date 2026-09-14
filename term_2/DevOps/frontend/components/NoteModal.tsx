import React, { useState, useEffect } from 'react';
import { Note, Area } from '../types';

interface NoteModalProps {
  note: Note | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: (note: Note) => void;
  areas: Area[];
}

const NoteModal: React.FC<NoteModalProps> = ({ note, isOpen, onClose, onSave, areas }) => {
  const [formData, setFormData] = useState<Partial<Note>>({});

  useEffect(() => {
    if (note) {
        setFormData({ ...note });
    }
  }, [note]);

  if (!isOpen || !note) return null;

  const handleSave = () => {
      if (formData.title) {
          // Create a clean copy of formData to ensure no unexpected properties are passed
          const cleanFormData = {
              id: note.id,
              title: formData.title,
              content: formData.content,
              area_id: formData.area_id,
          };
          onSave(cleanFormData as Note);
          onClose();
      }
  };

  const handleChange = (field: keyof Note, value: any) => {
      setFormData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="fixed inset-0 bg-gray-900/60 backdrop-blur-sm z-40 flex items-center justify-center p-4 sm:p-6">
      <div className="bg-white dark:bg-[#101622] w-full max-w-3xl h-[85vh] rounded-xl shadow-2xl flex flex-col border border-gray-200 dark:border-gray-700 animate-fade-in-up overflow-hidden">
        
        {/* Header */}
        <div className="px-8 pt-6 pb-2 flex-none">
            <div className="flex justify-between items-start mb-4">
                <div className="flex items-center space-x-2 text-sm text-gray-500 dark:text-gray-400">
                     <div className="flex items-center space-x-1">
                        <span className="material-icons text-base">folder_open</span>
                         <select 
                            value={formData.area_id || ''} 
                            onChange={(e) => handleChange('area_id', parseInt(e.target.value, 10))}
                            className="bg-transparent border-none text-sm font-medium focus:ring-0 cursor-pointer p-0 text-gray-500 dark:text-gray-400"
                        >
                            {areas.map(area => (
                                <option key={area.id} value={area.id}>{area.name}</option>
                            ))}
                        </select>
                    </div>
                    {formData.updated_at && (
                        <>
                            <span className="text-gray-300 dark:text-gray-600">/</span>
                            <span className="text-xs">Last edited {new Date(formData.updated_at).toLocaleDateString()}</span>
                        </>
                    )}
                </div>
                <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800">
                    <span className="material-icons">close</span>
                </button>
            </div>
            <input 
                className="text-3xl font-bold text-gray-900 dark:text-white bg-transparent border-none p-0 focus:ring-0 w-full placeholder-gray-300 outline-none" 
                value={formData.title || ''}
                onChange={(e) => handleChange('title', e.target.value)}
                placeholder="Untitled Note"
            />
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto px-8 py-4 custom-scrollbar">
            <textarea 
                className="w-full h-full resize-none bg-transparent border-none focus:ring-0 text-gray-600 dark:text-gray-300 leading-relaxed outline-none"
                value={formData.content || ''}
                onChange={(e) => handleChange('content', e.target.value)}
                placeholder="Start writing..."
            ></textarea>
        </div>

        {/* Footer */}
        <div className="px-8 py-4 border-t border-gray-100 dark:border-gray-800 bg-gray-50 dark:bg-gray-900/50 flex justify-end">
            <button onClick={handleSave} className="bg-primary hover:bg-primary-hover text-white px-6 py-2 rounded-lg font-medium text-sm shadow-md transition-all">
                Done
            </button>
        </div>
      </div>
    </div>
  );
};

export default NoteModal;
